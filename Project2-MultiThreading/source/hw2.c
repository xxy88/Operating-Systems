#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 
#define LENGTH (COLUMN - 1)
#define LOGSPEED 100000


struct Node {
	int x, y;
} frog;


char map[ROW+10][COLUMN] ; 

// 0: in process, 1: win, 2: lose, 3: quit
int status = 0 ;
// contain the length of each log
int log_lengths[ROW] ;
// contain the starting position of each log
int log_start[ROW] ;

pthread_mutex_t map_mutex;
pthread_mutex_t frog_mutex;

/*  Print the map on the screen  */
void display_map(){
	system("clear");
	for(int i = 0; i <= ROW; ++i)	
		puts( map[i] );
}

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

/*  Check game's status  */
void check_status(){
	if ((frog.y == 0) || (frog.y == LENGTH-1) || (map[frog.x][frog.y] == ' ') || (frog.x > ROW)) {
		status = 2;
	}
	if (frog.x == 0) {
		status = 1;
	}
}

char* get_item(char* arr, int index) {
	return &arr[(index + LENGTH) % LENGTH];
}

void* logs_move(void* t) {
	long tid = (long)t;
	while (status == 0) {
		usleep(LOGSPEED);
		pthread_mutex_lock(&map_mutex);
		char* row = map[tid];
		int* start = &log_start[tid];
		const int log_len = log_lengths[tid];
		// direction: if tid is odd: left, even: right
		if (tid % 2 == 1) {
			for (int i = *start; i <= *start + log_len; i++) {
				*get_item(row, i - 1) = *get_item(row, i);
			}
			*start = (*start - 1 + LENGTH) % LENGTH;
		} else {
			for (int i = *start + log_len; i >= *start; i--) {
				*get_item(row, i) = *get_item(row, i - 1);
			}
			*start = (*start + 1) % LENGTH;
		}
		// update position of frog
		if (frog.x == tid){
			pthread_mutex_lock(&frog_mutex);
			frog.y += (tid % 2) ? -1 : 1;
			pthread_mutex_unlock(&frog_mutex);
		}
		// check whether frog moves to left or right boundaries with log
		check_status();

		display_map();
		pthread_mutex_unlock(&map_mutex);
	}
	pthread_exit(NULL);
}

void* frog_move(void* t) {
	while (status == 0) {
		check_status();

		// if user hit some keyboard, catch action
		if (kbhit()){
			/*  Check keyboard hits, to change frog's position or quit the game. */
			char option = getchar();

			pthread_mutex_lock(&map_mutex);
			pthread_mutex_lock(&frog_mutex);
			map[frog.x][frog.y] = (frog.x == ROW) ? '|' : '=';
			switch (option)
			{
			case 'A': case 'a':
				frog.y --;
				break;
			case 'D': case 'd':
				frog.y ++;
				break;
			case 'W': case 'w':
				frog.x --;
				break;
			case 'S': case 's':
				frog.x ++;
				break;
			case 'Q': case 'q':
				status = 3;
				break;
			}
			check_status();
			if (status == 0) {
				map[frog.x][frog.y] = '0';
			}
			pthread_mutex_unlock(&frog_mutex);

			display_map();
			pthread_mutex_unlock(&map_mutex);
		}
	}
	pthread_exit(NULL);
}

// initialize log positions before game starts
void init_logs(){
	for (int i = 1; i < ROW; i++){
		int log_length = rand() % 10 + 10;
		log_lengths[i] = log_length;
		int log_start_pos = rand() % (COLUMN-1);
		log_start[i] = log_start_pos;
		for (int j = 0; j < log_length; j++){
			map[i][(log_start_pos + j) % (COLUMN-1)] = '=';
		}
	}
}

/*  Display the output for user: win, lose or quit.  */
void display_outputs(){
	system("clear");
	if (status == 1){
		printf("You win the game!!\n");
	}
	else if (status == 2){
		printf("You lose the game!!\n");
	}
	else if (status == 3){
		printf("You exit the game.\n");
	}
}

int main( int argc, char *argv[] ){
	pthread_t threads_logs[ROW];
	pthread_t thread_frog;
	pthread_mutex_init(&map_mutex, NULL);
	pthread_mutex_init(&frog_mutex, NULL);

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	frog.x = ROW;
	frog.y = (COLUMN-1) / 2;
	map[frog.x][frog.y] = '0' ; 

	//Print the map into screen
	init_logs();
	for(int i = 0; i <= ROW; ++i)	
		puts( map[i] );
	
	/*  Create pthreads for wood move and frog control.  */
	for (long i = 1; i < ROW; i++){
		if (pthread_create(&threads_logs[i], NULL, logs_move, (void *)i)){
			perror("Error to create thread!");
			exit(1);
		}
	}
	if (pthread_create(&thread_frog, NULL, frog_move, NULL)) {
		perror("Error to create thread!");
		exit(1);
	}
	
	// wait for other threads to end
	for (long i = 1; i < ROW; i++){
		if (pthread_join(threads_logs[i], NULL)){
			perror("Error to join thread!");
			exit(1);
		}
	}
	if (pthread_join(thread_frog, NULL)){
		perror("Error to join thread!");
		exit(1);
	}

	/*  Display the output for user: win, lose or quit.  */
	display_outputs();

	pthread_mutex_destroy(&map_mutex);
	pthread_mutex_destroy(&frog_mutex);

	return 0;
}
