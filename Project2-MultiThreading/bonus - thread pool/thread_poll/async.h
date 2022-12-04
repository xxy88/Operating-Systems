#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>


typedef struct my_task {
  void (*handler)(int);
  int args;
  struct my_task *next;
} my_task_t;

typedef struct my_task_queue {
  int task_num;
  my_task_t *head;
  my_task_t *tail;
} my_task_queue_t;

typedef struct my_thread_pool {
  int thread_num;
  pthread_t *threads_arr;
  my_task_queue_t *task_queue;
} my_thread_pool_t;


void async_init(int);
void async_run(void (*fx)(int), int args);
void add_task(my_task_t *task, my_task_queue_t *queue);
my_task_t* get_task(my_task_queue_t *queue);
void * thread_run(void *args);

#endif
