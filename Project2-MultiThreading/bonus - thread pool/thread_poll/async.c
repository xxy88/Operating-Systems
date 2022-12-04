#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <stdio.h>
#include "async.h"
#include "utlist.h"

static pthread_mutex_t task_queue_mutex;
static pthread_cond_t thread_cv;
my_thread_pool_t *pool;


void async_init(int num_threads) {
    pthread_mutex_init(&task_queue_mutex, NULL);
    pthread_cond_init(&thread_cv, NULL);

    pool = malloc(sizeof(my_thread_pool_t));
    pool->task_queue = malloc(sizeof(my_task_queue_t));
    pool->thread_num = num_threads;
    pool->threads_arr = malloc(sizeof(pthread_t) * num_threads);
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&pool->threads_arr[i], NULL, thread_run, NULL)){
            perror("Error to create thread!");
			exit(1);
        }
    }
}

void * thread_run(void *args) {
    while (1) {
        pthread_mutex_lock(&task_queue_mutex);
        while (pool->task_queue->task_num == 0) {
            pthread_cond_wait(&thread_cv, &task_queue_mutex);
        }
        my_task_t *task = get_task(pool->task_queue);
        pthread_mutex_unlock(&task_queue_mutex);
        if (task != NULL) {
            task->handler(task->args);
            free(task);
        }
    }
    pthread_exit(NULL);
}


// add a newly submitted task to the end of task queue
void add_task(my_task_t *task, my_task_queue_t *queue) {
    pthread_mutex_lock(&task_queue_mutex);
    if (queue->task_num == 0) {
        queue->head = task;
        queue->tail = task;
    } else {
        queue->tail->next = task;
        queue->tail = task;
    }
    queue->task_num++;
    pthread_mutex_unlock(&task_queue_mutex);
}

// get a task from the head of task queue
my_task_t* get_task(my_task_queue_t *queue) {
    if (queue->task_num == 0) {
        return NULL;
    }
    my_task_t *task = queue->head;
    queue->head = queue->head->next;
    queue->task_num--;
    return task;
}

void async_run(void (*handler)(int), int args) {
    my_task_t *task = malloc(sizeof(my_task_t));
    task->handler = handler;
    task->args = args;
    task->next = NULL;
    add_task(task, pool->task_queue);
    pthread_cond_signal(&thread_cv);
}
