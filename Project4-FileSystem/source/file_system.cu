#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "file_system.h"

__device__ __managed__ u16 gtime = 0;

#define NAME_OFFSET 0
#define SIZE_OFFSET 20
#define CREATION_OFFSET 22
#define MODIFICATION_OFFSET 24 

#define VCB_BITMASK(index) 0x80 >> (index) % 8

#define CHECK_NULL(ptr)                  \
    if ((ptr) == nullptr) {              \
        printf("Error: null pointer\n"); \
        assert(false);                   \
    }

__device__ void str_copy(uchar* dest, const uchar* src) {
    while (*dest++ = *src++)
        ;
}

__device__ void str_copy(char* dest, const char* src) {
    while (*dest++ = *src++)
        ;
}

__device__ uchar* link_fcb_ptr_(FileSystem* fs, int index) {
    if (index >= fs->FCB_ENTRIES) {
        printf("Error: index out of range: %d\n", index);
        assert(false);
    }
    return &fs->volume[fs->SUPERBLOCK_SIZE + index * fs->FCB_SIZE];
}

__device__ char* fcb_get_name(FileSystem* fs, int index) {
    return (char*)((link_fcb_ptr_(fs, index) + NAME_OFFSET));
}

__device__ void fcb_set_name(FileSystem* fs, int index, const char* name) {
    str_copy(fcb_get_name(fs, index), name);
}

__device__ u16* fcb_size_ptr(FileSystem* fs, int index) {
    return (u16*)((link_fcb_ptr_(fs, index) + SIZE_OFFSET));
}

__device__ u16* fcb_create_time_ptr(FileSystem* fs, int index) {
    return (u16*)((link_fcb_ptr_(fs, index) + CREATION_OFFSET));
}

__device__ u16* fcb_modify_time_ptr(FileSystem* fs, int index) {
    return (u16*)((link_fcb_ptr_(fs, index) + MODIFICATION_OFFSET));
}

__device__ uchar* fcb_get_content(FileSystem* fs, int index) {
    return &fs->volume[fs->FILE_BASE_ADDRESS + index * fs->STORAGE_BLOCK_SIZE];
}

__device__ int size_2_blocks(FileSystem* fs, int size) {
    const u16 quotient = size / fs->STORAGE_BLOCK_SIZE;
    const u16 remainder = size % fs->STORAGE_BLOCK_SIZE ? 1 : 0;
    return quotient + remainder;
}

__device__ int fcb_get_start_block(FileSystem* fs, int index) {
    return index * (fs->MAX_FILE_SIZE / fs->MAX_FILE_NUM / fs->STORAGE_BLOCK_SIZE);
}

__device__ void vcb_fill(FileSystem* fs, int index) {
    int start_block = fcb_get_start_block(fs, index);
    int num_blocks = size_2_blocks(fs, *fcb_size_ptr(fs, index));
    int end_block = start_block + num_blocks;
    for (int i = start_block; i < end_block; ++i) {
        fs->volume[index / 8] |= VCB_BITMASK(index);
    }
}

__device__ void vcb_free(FileSystem* fs, int index) {
    int start_block = fcb_get_start_block(fs, index);
    int num_blocks = size_2_blocks(fs, *fcb_size_ptr(fs, index));
    int end_block = start_block + num_blocks;
    for (int i = start_block; i < end_block; ++i) {
        fs->volume[index / 8] &= ~VCB_BITMASK(index);
    }
}

__device__ void sort(FileSystem* fs, int* fcbs, const int size, int op) {
    for (int i = 0; i < size - 1; ++i) {
        int max_idx = i;
        for (int j = i + 1; j < size; ++j) {
            bool need_swap = false;
            switch (op) {
            case LS_D:
                need_swap = (*fcb_modify_time_ptr(fs, fcbs[j]) > *fcb_modify_time_ptr(fs, fcbs[max_idx]));
                break;
            case LS_S:
                if (*fcb_size_ptr(fs, fcbs[j]) > *fcb_size_ptr(fs, fcbs[max_idx]) ||
                    *fcb_size_ptr(fs, fcbs[j]) == *fcb_size_ptr(fs, fcbs[max_idx]) &&
                    *fcb_create_time_ptr(fs, fcbs[j]) < *fcb_create_time_ptr(fs, fcbs[max_idx])) 
                    need_swap = true;
                break;
            default:
                printf("Invalid op for sort: %d\n", op);
                return;
            }
            if (need_swap) {
                max_idx = j;
            }
        }
        if (max_idx != i) {
            int temp = fcbs[i];
            fcbs[i] = fcbs[max_idx];
            fcbs[max_idx] = temp;
        }
    }
}

__device__ bool str_equal(const char* s1, const char* s2) {
    int i = 0;
    while(s1[i] || s2[i]) {
        if (s1[i] != s2[i]) {
            return false;
        }
        i++;
    }
    return true;
}

__device__ void fs_init(FileSystem* fs, uchar* volume, int SUPERBLOCK_SIZE, int FCB_SIZE,
                        int FCB_ENTRIES, int VOLUME_SIZE, int STORAGE_BLOCK_SIZE,
                        int MAX_FILENAME_SIZE, int MAX_FILE_NUM, int MAX_FILE_SIZE,
                        int FILE_BASE_ADDRESS) {
    // init variables
    fs->volume = volume;
    fs->file_num = 0;

    // init constants
    fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
    fs->FCB_SIZE = FCB_SIZE;
    fs->FCB_ENTRIES = FCB_ENTRIES;
    fs->STORAGE_SIZE = VOLUME_SIZE;
    fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
    fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
    fs->MAX_FILE_NUM = MAX_FILE_NUM;
    fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
    fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
}


__device__ u32 fs_open(FileSystem* fs, char* s, int op) {
    /* Implement open operation here */
    int result = -1;
    for (int i = 0; i < fs->MAX_FILE_NUM; ++i) {
        char* filename = fcb_get_name(fs, i);
        CHECK_NULL(filename)
        if (filename[0] == '\0' && result == -1)
            result = i;
        else if (str_equal(filename, s))
            return i;
    }
    switch (op) {
    case G_READ:
        printf("Error reading: No such file %s\n", s);
        return 0xFFFFFFFFu;
    case G_WRITE:
        if (result != -1) {
            fcb_set_name(fs, result, s);
            *fcb_size_ptr(fs, result) = 0;
            *fcb_create_time_ptr(fs, result) = gtime;
            *fcb_modify_time_ptr(fs, result) = gtime;
            ++gtime;
            fs->file_num++;
            return result;
        }
        else {
            printf("Error: Unable to create a new file, no space available\n");
            return 0xFFFFFFFFu;
        }
    default:
        printf("Invalid op: %d\n", op);
        return 0xFFFFFFFFu;
    }
}

__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, u32 fp) {
    /* Implement read operation here */
    if (fp != 0xFFFFFFFFu) str_copy(output, fcb_get_content(fs, fp));
}

__device__ u32 fs_write(FileSystem* fs, uchar* input, u32 size, u32 fp) {
    /* Implement write operation here */
    if (fp != 0xFFFFFFFFu) {
        *fcb_size_ptr(fs, fp) = size;
        *fcb_modify_time_ptr(fs, fp) = gtime++;
        str_copy(fcb_get_content(fs, fp), input);
        vcb_fill(fs, fp);
    }
}

__device__ void fs_gsys(FileSystem* fs, int op) {
    /* Implement LS_D and LS_S operation here */
    int* fcbs = nullptr;
    cudaMalloc(&fcbs, sizeof(int) * fs->file_num);
    int index = 0;
    for (int i = 0; i < fs->MAX_FILE_NUM; ++i) {
        if (fcb_get_name(fs, i)[0] != 0) fcbs[index++] = i;
    }
    assert(index == fs->file_num);
    sort(fs, fcbs, fs->file_num, op);
    switch (op) {
    case LS_D:
        printf("===sort by modified time===\n");
        for (int i = 0; i < fs->file_num; i++) {
            printf("%s\n", fcb_get_name(fs, fcbs[i]));
        }
        break;
    case LS_S:
        printf("===sort by file size===\n");
        for (int i = 0; i < fs->file_num; i++) {
            printf("%s %d\n", fcb_get_name(fs, fcbs[i]), *fcb_size_ptr(fs, fcbs[i]));
        }
        break;
    default:
        printf("Invalid op: %d\n", op);
    }
    cudaFree(fcbs);
}

__device__ void fs_gsys(FileSystem* fs, int op, char* s) {
    /* Implement rm operation here */
    if (op != RM) {
        printf("Invalid op: %d, should be rm\n", op);
        return;
    }
    for (int i = 0; i < fs->MAX_FILE_NUM; ++i) {
        char* name = fcb_get_name(fs, i);
        if (str_equal(name, s)) {
            name[0] = '\0';
            vcb_free(fs, i);
            fs->file_num--;
            return;
        }
    }
}
