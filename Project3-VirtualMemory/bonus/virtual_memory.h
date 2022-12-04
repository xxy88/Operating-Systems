#pragma once

#include <cinttypes>
#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned char uchar;
typedef uint32_t u32;

struct VirtualMemory {
    int* thd_id;
    uchar* buffer;
    uchar* storage;
    u32* invert_page_table;

    uchar* freespace_map;
    u32* swap_table;
    uchar* temp_buffer;

    int* pagefault_num_ptr;

    int PAGESIZE;
    int INVERT_PAGE_TABLE_SIZE;
    int PHYSICAL_MEM_SIZE;
    int STORAGE_SIZE;
    int PAGE_ENTRIES;

    u32 mru;
    u32 lru;
    u32 second_lru;

    int num_entries;
};

__device__ void vm_init(VirtualMemory* vm, uchar* buffer, uchar* storage, u32* invert_page_table,
                        u32* swap_table, uchar* storage_bitmap, uchar* swap_buffer,
                        int* pagefault_num_ptr, int page_size, int invert_page_table_size,
                        int physical_mem_size, int storage_size, int page_entries, int* thd_id);
__device__ uchar vm_read(VirtualMemory* vm, u32 addr);
__device__ void vm_write(VirtualMemory* vm, u32 addr, uchar value);
__device__ void vm_snapshot(VirtualMemory* vm, uchar* results, int offset, int input_size);
