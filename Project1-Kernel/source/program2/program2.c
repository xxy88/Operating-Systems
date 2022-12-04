#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");

#define WEXITSTATUS(status) ((status & 0xff00) >> 8)
#define WTERMSIG(status) (status & 0x7f)
#define WSTOPSIG(status) (WEXITSTATUS(status))
#define WIFEXITED(status) (WTERMSIG(status) == 0)
#define WIFSIGNALED(status) (((signed char)(((status & 0x7f) + 1) >> 1)) > 0)
#define WIFSTOPPED(status) (((status)&0xff) == 0x7f)

char *signals[] = { "SIGHUP",  "SIGINT",  "SIGQUIT", "SIGILL",	"SIGTRAP",
		    "SIGABRT", "SIGBUS",  "SIGFPE",  "SIGKILL", "SIGUSR1",
		    "SIGSEGV", "SIGUSR2", "SIGPIPE", "SIGALRM", "SIGTERM" };

struct wait_opts {
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;
	struct waitid_info *wo_info;
	int wo_stat;
	struct rusage *wo_rusage;
	wait_queue_entry_t child_wait;
	int notask_error;
};

extern long do_wait(struct wait_opts *wo);
extern int do_execve(struct filename *filename,
		     const char __user *const __user *__argv,
		     const char __user *const __user *__envp);

int my_wait(pid_t pid)
{
	int a;
	int status;
	struct wait_opts wo;
	struct pid *wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);

	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WEXITED | WUNTRACED;
	wo.wo_info = NULL;
	wo.wo_rusage = NULL;

	a = do_wait(&wo);
	status = wo.wo_stat;

	put_pid(wo_pid);

	return status;
}

// implement execution function
int my_exec(void)
{
	// execute the test in child process
	int result;
	struct filename *my_filename;
	const char path[] = "/tmp/test";

	printk("[program2] : child process");
	my_filename = getname_kernel(path);
	result = do_execve(my_filename, NULL, NULL);

	if (!result)
		return 0;
	do_exit(result);
}

void check_status(int status)
{
	if (WIFEXITED(status)) {
		printk("[program2] : Normal termination with EXIT STATUS = %d\n",
		       WEXITSTATUS(status));
	} else if (WIFSTOPPED(status)) {
		printk("[program2] : get SIGSTOP signal\n");
		printk("[program2] : child process terminated\n");
		printk("[program2] : The return signal is %d\n",
		       WSTOPSIG(status));
	} else if (WIFSIGNALED(status)) {
		printk("[program2] : get %s signal\n",
		       signals[WTERMSIG(status) - 1]);
		printk("[program2] : child process terminated\n");
		printk("[program2] : The return signal is %d\n",
		       WTERMSIG(status));
	} else {
		printk("[program2] : child process continued\n");
	}
}

// implement fork function
int my_fork(void *argc)
{
	//set default sigaction for current process
	int i;
	pid_t pid;
	int status;
	struct k_sigaction *k_action;
	struct kernel_clone_args kargs = { .flags = SIGCHLD,
					   .stack = (unsigned long)&my_exec,
					   .stack_size = 0,
					   .parent_tid = NULL,
					   .child_tid = NULL,
					   .tls = 0 };
	;
	k_action = &current->sighand->action[0];
	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	/* fork a process using kernel_clone or kernel_thread */
	pid = kernel_clone(&kargs);
	printk("[program2] : The child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n",
	       (int)current->pid);

	/* execute a test program in child process */

	/* wait until child process terminates */
	msleep(50);
	status = my_wait(pid);
	check_status(status);

	return 0;
}

static int __init program2_init(void)
{
	struct task_struct *my_kthread;
	printk("[program2] : Module_init Xinyu Xie 119020059\n");
	printk("[program2] : module_init create kthread start");

	/* create a kernel thread to run my_fork */
	my_kthread = kthread_create(&my_fork, NULL, "my_kthread");

	// wake up new thread if ok
	if (!IS_ERR(my_kthread)) {
		printk("[program2] : module_init kthread start");
		wake_up_process(my_kthread);
	}

	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : Module_exit./my\n");
}

module_init(program2_init);
module_exit(program2_exit);
