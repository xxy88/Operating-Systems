#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>
#include <string.h>

int main(int argc, char *argv[])
{
	pid_t pid;
	int status;
	char *signals[] = { "SIGHUP",  "SIGINT",  "SIGQUIT", "SIGILL",
			    "SIGTRAP", "SIGABRT", "SIGBUS",  "SIGFPE",
			    "SIGKILL", "SIGUSR1", "SIGSEGV", "SIGUSR2",
			    "SIGPIPE", "SIGALRM", "SIGTERM" };
	printf("Process start to fork\n");

	/* fork a child process */
	pid = fork();

	if (pid == -1) {
		perror("fork");
		exit(1);
	} else {
		// child process
		if (pid == 0) {
			printf("I'm the Child Process, my pid = %d\n",
			       getpid());
			printf("Child process start to execute test program:\n");

			/* execute test program */
			char *arg[argc];
			for (int i = 0; i < argc - 1; ++i) {
				arg[i] = argv[i + 1];
			}
			arg[argc - 1] = NULL;

			execve(arg[0], arg, NULL);

			perror("execve");
			exit(1);
		}

		// parent process
		else {
			printf("I'm the Parent Process, my pid = %d\n",
			       getpid());
			/* wait for child process terminates */
			waitpid(pid, &status, WUNTRACED);
			printf("Parent process receives SIGCHLD signal\n");

			/* check child process'  termination status */
			if (WIFEXITED(status)) {
				printf("Normal termination with EXIT STATUS = %d\n",
				       WEXITSTATUS(status));
			} else if (WIFSIGNALED(status)) {
				printf("child process get %s signal\n",
				       signals[WTERMSIG(status) - 1]);
			} else if (WIFSTOPPED(status)) {
				printf("child process get SIGSTOP signal\n");
			} else {
				printf("child process continued\n");
			}
			exit(0);
		}
	}

	return 0;
}
