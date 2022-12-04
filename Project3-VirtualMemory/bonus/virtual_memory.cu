#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include "virtual_memory.h"

#include <device_launch_parameters.h>

#define PPN_MASK (0x3FFu)
#define VPN_MASK (0x1FFFu)
#define OFFSET_MASK (0x1Fu)

#define GET_PAGE_NUMBER(address) ((address) >> 5)
#define GET_OFFSET(address) ((address)&OFFSET_MASK)
#define IS_VALID(entry) ((entry) >> 31 == 1)
#define GET_VPN_FROM_ENTRY(entry) ((entry)&VPN_MASK)
#define GET_NEXT_LRU_FROM_ENTRY(entry) (((entry) >> 21) & PPN_MASK)
#define COMBINE_ADDR(page_number, offset) (((page_number) << 5) + (offset))

// set `next LRU` entry of a page table entry to given value, assuming the entry is valid
#define SET_ENTRY_NEXT_LRU(entry, idx) \
    { (entry) = ((entry) & ~(PPN_MASK << 21)) | ((idx) << 21); }
// set VPN
#define SET_ENTRY_VPN(entry, vpn) \
    { (entry) = ((entry) & ~VPN_MASK) + (vpn); }

// first PAGE_ENTRIES entries are for page table
__device__ void init_invert_page_table(VirtualMemory* vm) {
    vm->mru = 0x3ffu;
    vm->lru = 0x3ffu;
    vm->second_lru = 0x3ffu;
    vm->num_entries = 0;
}

__device__ void vm_init(VirtualMemory* vm, uchar* buffer, uchar* storage, u32* invert_page_table,
                        u32* swap_table, uchar* storage_bitmap, uchar* swap_buffer,
                        int* pagefault_num_ptr, int page_size, int invert_page_table_size,
                        int physical_mem_size, int storage_size, int page_entries, int* thd_id) {
    // initialize variables
    vm->thd_id = thd_id;
    *vm->thd_id = threadIdx.x;
    vm->buffer = buffer;
    vm->storage = storage;
    vm->invert_page_table = invert_page_table;
    vm->pagefault_num_ptr = pagefault_num_ptr;
    vm->swap_table = swap_table;
    vm->freespace_map = storage_bitmap;
    vm->temp_buffer = swap_buffer;

    // initialize constants
    vm->PAGESIZE = page_size;
    vm->INVERT_PAGE_TABLE_SIZE = invert_page_table_size;
    vm->PHYSICAL_MEM_SIZE = physical_mem_size;
    vm->STORAGE_SIZE = storage_size;
    vm->PAGE_ENTRIES = page_entries;

    // before first vm_write or vm_read
    init_invert_page_table(vm);

    // initialize swap table
    for (int i = 0; i < (physical_mem_size + storage_size) / page_size; ++i) {
        vm->swap_table[i] = 0xFFFFFFFFu;
    }
}

__device__ void clear(VirtualMemory* vm) {
    // clear page table
    for (int i = 0; i < vm->PAGE_ENTRIES; ++i) {
        vm->invert_page_table[i] = 0;
    }
    vm->mru = 0x3ffu;
    vm->lru = 0x3ffu;
    vm->second_lru = 0x3ffu;
    vm->num_entries = 0;

    // clear swap table
    for (int i = 0; i < (vm->PHYSICAL_MEM_SIZE + vm->STORAGE_SIZE) / vm->PAGESIZE; ++i) {
        vm->swap_table[i] = 0xFFFFFFFFu;
    }
    for (int i = 0; i < vm->STORAGE_SIZE / vm->PAGESIZE; ++i) {
        vm->freespace_map[i] = 0;
    }
}

// bring desired page from 'disk' into 'memory' (and evict existing pages if necessary)
__device__ u32 swap_from_disk(VirtualMemory* vm, u32 vpn, bool is_read) {
    u32 evict_idx = vm->lru;
    u32 evict_phy_addr = COMBINE_ADDR(evict_idx, 0);
    u32 evict_vpn = GET_VPN_FROM_ENTRY(vm->invert_page_table[evict_idx]);

    if (evict_idx != vm->mru) {
        SET_ENTRY_NEXT_LRU(vm->invert_page_table[vm->second_lru],
                           GET_NEXT_LRU_FROM_ENTRY(vm->invert_page_table[evict_idx]))
        SET_ENTRY_NEXT_LRU(vm->invert_page_table[evict_idx], vm->mru)
        vm->mru = evict_idx;
        
        if (vm->lru == evict_idx) { vm->lru = vm->second_lru; }
    }

    // swap out
    for (int i = 0; i < vm->PAGESIZE; ++i) {
        vm->temp_buffer[i] = vm->buffer[evict_phy_addr + i];
    }

    // swap in
    SET_ENTRY_VPN(vm->invert_page_table[evict_idx], vpn)
    if (vm->swap_table[vpn] != 0xFFFFFFFFu) {
        for (int i = 0; i < vm->PAGESIZE; ++i) {
            vm->buffer[evict_phy_addr + i] = vm->storage[vm->swap_table[vpn] + i];
        }
        vm->freespace_map[GET_PAGE_NUMBER(vm->swap_table[vpn])] = 0;
        vm->swap_table[vpn] = 0xFFFFFFFFu;
    }
    else if (is_read) {
        printf("No result available for read.\n");
        assert(false);
    }

    // write temp buffer back to storage
    u32 slot_num = 0;
    while (vm->freespace_map[slot_num] != 0) {
        slot_num++;
    }
    vm->freespace_map[slot_num] = 1;
    vm->swap_table[evict_vpn] = COMBINE_ADDR(slot_num, 0);
    for (int i = 0; i < vm->PAGESIZE; ++i) {
        vm->storage[COMBINE_ADDR(slot_num, 0) + i] = vm->temp_buffer[i];
    }
    return evict_idx;
}

__device__ u32 get_physical_address(VirtualMemory* vm, u32 virtual_address, bool is_read) {
    // check thread
    if (threadIdx.x != *vm->thd_id) {
        clear(vm);
        *vm->thd_id = threadIdx.x;
    }

    u32 ppn;
    bool addr_in_memory = false;
    // scan through the LRU list to check whether the page is in memory
    if (vm->num_entries != 0) {
        u32 previous_idx = 0xFFFFFFFF;
        u32 current_idx = vm->mru;
        while (previous_idx != vm->lru) {
            if (GET_VPN_FROM_ENTRY(vm->invert_page_table[current_idx]) ==
                GET_PAGE_NUMBER(virtual_address)) {
                ppn = current_idx;
                // modify the LRU list
                if (current_idx != vm->mru) {
                    SET_ENTRY_NEXT_LRU(vm->invert_page_table[previous_idx],
                                       GET_NEXT_LRU_FROM_ENTRY(vm->invert_page_table[current_idx]))
                    SET_ENTRY_NEXT_LRU(vm->invert_page_table[current_idx], vm->mru)
                    vm->mru = current_idx;
                    if (vm->lru == current_idx) { vm->lru = previous_idx; }
                }
                addr_in_memory = true;
                break;
            }
            // next iteration
            vm->second_lru = previous_idx;
            previous_idx = current_idx;
            current_idx = GET_NEXT_LRU_FROM_ENTRY(vm->invert_page_table[current_idx]);
        }
        if (!addr_in_memory) (*vm->pagefault_num_ptr)++;
    }

    if (!addr_in_memory || !IS_VALID(vm->invert_page_table[ppn])) {
        if (vm->num_entries < vm->PAGE_ENTRIES) {
            if (is_read) {
                printf("No result available for read.\n");
                assert(false);
            }
            else {
                // first time
                if (vm->num_entries == 0) {
                    (*vm->pagefault_num_ptr)++;
                    vm->lru = 0;
                }
                vm->invert_page_table[vm->num_entries] =
                    (1u << 31) + (vm->mru << 21) + (GET_PAGE_NUMBER(virtual_address));
                vm->mru = vm->num_entries++;
                ppn = vm->mru;
            }
        }
        else { ppn = swap_from_disk(vm, GET_PAGE_NUMBER(virtual_address), is_read); }
    }
    return COMBINE_ADDR(ppn, GET_OFFSET(virtual_address));
}

__device__ uchar vm_read(VirtualMemory* vm, u32 addr) {
    return vm->buffer[get_physical_address(vm, addr, true)];
}

__device__ void vm_write(VirtualMemory* vm, u32 addr, uchar value) {
    vm->buffer[get_physical_address(vm, addr, false)] = value;
}

__device__ void vm_snapshot(VirtualMemory* vm, uchar* results, int offset, int input_size) {
    /* Complete snapshot function together with vm_read to
       load elements from data to result buffer */
    for (int i = 0; i < input_size; i++) {
        results[i] = vm_read(vm, i + offset);
    }
}
