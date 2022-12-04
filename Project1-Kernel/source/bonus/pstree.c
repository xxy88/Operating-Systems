#include <dirent.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <sys/types.h>

#define PROC_BASE "/proc"

typedef struct proc {
	char name[256];
	pid_t pid;
	pid_t ppid;
	struct proc *parent;
	struct child *children;
	struct proc *next;
} PROC;

typedef struct child {
	PROC *proc;
	int rep;
	struct child *next;
} CHILD;

// pstree formatter
typedef struct format {
	const char *single;
	const char *first;
	const char *branch;
	const char *vertical;
	const char *last;
	const char *blank;
} format;

static format formatter = { "---", "-+-", " |-", " | ", " `-", "   " };

// command line options
static int compact = 1, pids = 0, hide_threads = 0, trunc = 0, ascii = 1;
static int root_pid = 1;

static PROC *head;
static PROC *roots[32];
int roots_count = 0;

char *strip(char *str)
{
	char *end;
	while (isspace((unsigned char)*str))
		str++;
	if (*str == '\0')
		return str;
	end = str + strlen(str) - 1;
	while (end > str && isspace((unsigned char)*end))
		end--;
	end[1] = '\0';
	return str;
}

PROC *add_node(const char *comm, pid_t pid, pid_t ppid, char is_thread)
{
	PROC *node;

	node = (PROC *)malloc(sizeof(PROC));
	if (node == NULL)
		exit(1);
	node->pid = pid;
	node->ppid = ppid;
	node->children = NULL;
	node->parent = NULL;
	if (is_thread)
		sprintf(node->name, "{%s}", comm);
	else
		strcpy(node->name, comm);
	node->next = head;
	head = node;

	return node;
}

PROC *find_node(pid_t pid)
{
	PROC *node;
	for (node = head; node != NULL; node = node->next) {
		if (node->pid == pid)
			return node;
	}
	return NULL;
}

void add_child(PROC *parent, PROC *child)
{
	CHILD *chld;
	chld = (CHILD *)malloc(sizeof(CHILD));
	if (chld == NULL)
		exit(1);
	chld->proc = child;
	chld->rep = 1;
	chld->next = parent->children;
	parent->children = chld;
	child->parent = parent;
}

void make_tree(void)
{
	/* from proc linked list, build a process tree */
	PROC *pnode;
	PROC *node;
	for (node = head; node != NULL; node = node->next) {
		pnode = find_node(node->ppid);
		if (pnode != NULL)
			add_child(pnode, node);
		else
			roots[roots_count++] = node;
	}
}

int treecmp(const PROC *a, const PROC *b)
{
	/* similar to strcmp, compare two trees' equality */
	CHILD *walk_a;
	CHILD *walk_b;
	if (a == NULL || b == NULL)
		return 1;
	if (strcmp(a->name, b->name) != 0)
		return 1;
	for (walk_a = a->children, walk_b = b->children; walk_a && walk_b;
	     walk_a = walk_a->next, walk_b = walk_b->next)
		if (treecmp(walk_a->proc, walk_b->proc))
			return 1;

	return walk_a || walk_b;
}

void combine_nodes(PROC *root)
{
	CHILD *child;
	CHILD *scan;
	CHILD *next;

	if (root == NULL)
		return;
	for (child = root->children; child != NULL; child = child->next)
		combine_nodes(child->proc);

	child = root->children;
	while (child != NULL) {
		scan = child;
		next = scan->next;
		while (next != NULL) {
			if (treecmp(child->proc, next->proc)) {
				scan = scan->next;
				next = next->next;
			} else {
				child->rep++;
				scan->next = next->next;
				free(next);
				next = scan->next;
			}
		}
		child = child->next;
	}
}

void dump_n_child(PROC *root, char *indent, int closing, char *comm);
void dump_one_child(PROC *root, char *indent, int closing, char *comm);

void dump_tree(PROC *root, char *indent, int rep, int closing)
{
	char comm[256];

	// if compact, combine children and add prefix
	if (compact) {
		combine_nodes(root);
		if (rep - 1)
			printf("%d*[", rep);
	}
	strcpy(comm, root->name);

	// dump pid
	if (pids)
		sprintf(comm, "%s(%d)", root->name, root->pid);

	printf("%s", comm);

	// recursive print
	if (root->children == NULL) {
		for (int i = 0; i < closing; ++i)
			printf("]");
		printf("\n");
	} else if (root->children->next == NULL) {
		dump_one_child(root, indent, closing, comm);
	} else {
		dump_n_child(root, indent, closing, comm);
	}
}

void dump_one_child(PROC *root, char *indent, int closing, char *comm)
{
	char new_indent[256];
	int rep = root->children->rep;
	printf("%s", formatter.single);
	sprintf(new_indent, "%s%s%*s", indent, formatter.blank,
		(int)strlen(comm), "");
	dump_tree(root->children->proc, new_indent, rep,
		  closing + (rep - 1) ? 1 : 0);
}

void dump_n_child(PROC *root, char *indent, int closing, char *comm)
{
	char new_indent[256];
	CHILD *scan;
	scan = root->children;
	printf("%s", formatter.first);
	sprintf(new_indent, "%s%*s%s", indent, (int)strlen(comm), "",
		formatter.vertical);
	dump_tree(scan->proc, new_indent, scan->rep,
		  closing + (scan->rep - 1) ? 1 : 0);

	for (scan = scan->next; scan->next != NULL; scan = scan->next) {
		printf("%s%*s%s", indent, (int)strlen(comm), "",
		       formatter.branch);
		sprintf(new_indent, "%s%*s%s", indent, (int)strlen(comm), "",
			formatter.vertical);
		dump_tree(scan->proc, new_indent, scan->rep,
			  closing + (scan->rep - 1) ? 1 : 0);
	}

	printf("%s%*s%s", indent, (int)strlen(comm), "", formatter.last);
	sprintf(new_indent, "%s%s%*s", indent, formatter.blank,
		(int)strlen(comm), "");
	dump_tree(scan->proc, new_indent, scan->rep,
		  closing + (scan->rep - 1) ? 1 : 0);
}

void add_thread(const char *dirname, const char *comm, pid_t proc_pid)
{
	char thread_filename[256];
	char buffer[256];
	int pid;
	FILE *file_ptr;
	char *key;
	char *value;

	sprintf(thread_filename, "%s/status", dirname);
	file_ptr = fopen(thread_filename, "r");
	if (file_ptr == NULL) {
		return;
	}

	while (fgets(buffer, sizeof(buffer), file_ptr) != NULL) {
		key = strtok(buffer, ":");
		value = strtok(NULL, ":");
		if (key != NULL && value != NULL) {
			key = strip(key);
			value = strip(value);
			if (strcmp(key, "Pid") == 0) {
				pid = atoi(value);
			}
		}
	}
	add_node(comm, pid, proc_pid, 1);

	fclose(file_ptr);
}

int add_proc(const char *dirname)
{
	char proc_filename[256];
	char buffer[256];
	char procname[256];
	char t_dirname[256];
	int pid;
	int ppid;
	char *key;
	char *value;
	FILE *file_ptr;
	DIR *dir_ptr;
	struct dirent *dir_entry;

	sprintf(proc_filename, "%s/status", dirname);
	file_ptr = fopen(proc_filename, "r");
	if (file_ptr == NULL) {
		return 1;
	}

	while (fgets(buffer, sizeof(buffer), file_ptr) != NULL) {
		key = strtok(buffer, ":");
		value = strtok(NULL, ":");
		if (key != NULL && value != NULL) {
			key = strip(key);
			value = strip(value);
			if (strcmp(key, "Pid") == 0) {
				pid = atoi(value);
			} else if (strcmp(key, "PPid") == 0) {
				ppid = atoi(value);
			} else if (strcmp(key, "Name") == 0) {
				strcpy(procname, value);
			}
		}
	}
	add_node(procname, pid, ppid, 0);
	fclose(file_ptr);

	// read a process's threads
	if (!hide_threads) {
		sprintf(t_dirname, "%s/task", dirname);
		dir_ptr = NULL;
		dir_entry = NULL;
		if ((dir_ptr = opendir(t_dirname)) == NULL) {
			exit(1);
		}
		while ((dir_entry = readdir(dir_ptr)) != NULL) {
			if (dir_entry->d_type == 4 &&
			    isdigit(dir_entry->d_name[0])) {
				if (atoi(dir_entry->d_name) == pid)
					continue;
				sprintf(proc_filename, "%s/%s", t_dirname,
					dir_entry->d_name);
				add_thread(proc_filename, procname, pid);
			}
		}
		closedir(dir_ptr);
	}

	return 0;
}

void read_procs()
{
	DIR *dir_ptr = NULL;
	struct dirent *dir_entry = NULL;
	char dirname[256];
	if ((dir_ptr = opendir(PROC_BASE)) == NULL)
		exit(1);

	// read proc info
	while ((dir_entry = readdir(dir_ptr)) != NULL) {
		if (dir_entry->d_type == 4 && isdigit(dir_entry->d_name[0])) {
			sprintf(dirname, "%s/%s", PROC_BASE, dir_entry->d_name);
			add_proc(dirname);
		}
	}
	free(dir_ptr);
}

void fix_orphans(PROC *root)
{
	PROC *node;
	for (int i = 0; i < roots_count; i++) {
		node = (PROC *)roots[i];
		if (node->ppid == 0)
			continue;
		if (node->parent == NULL) {
			add_child(root, node);
		}
	}
}

void parse_args(int argc, char **argv)
{
	struct option options[] = { { "ascii", 0, NULL, 'A' },
				    { "compact-not", 0, NULL, 'c' },
				    { "long", 0, NULL, 'l' },
				    { "show-pids", 0, NULL, 'p' },
				    { "hide-threads", 0, NULL, 'T' },
				    { 0, 0, 0, 0 } };

	int c;
	while ((c = getopt_long(argc, argv, "AcplT", options, NULL)) != -1) {
		switch (c) {
		case 'A':
			ascii = 1;
			break;
		case 'c':
			compact = 0;
			break;
		case 'p':
			pids = 1;
			compact = 0;
			break;
		case 'l':
			trunc = 0;
			break;
		case 'T':
			hide_threads = 1;
			break;
		}
	}
	if (optind == argc - 1) {
		if (isdigit(argv[optind][0])) {
			root_pid = atoi(argv[optind]);
		}
	}
}

PROC *find_root()
{
	PROC *root;
	root = find_node(root_pid);
	if (root_pid != 1) {
		if (root == NULL) {
			exit(1);
		}
	} else {
		fix_orphans(root);
	}
	return root;
}

int main(int argc, char **argv)
{
	PROC *root;

	// parse arguments
	parse_args(argc, argv);

	// read /proc to load process nodes
	read_procs();

	// build a process tree
	make_tree();

	// find the root node
	root = find_root();

	// dump the tree
	dump_tree(root, "", 1, 0);

	return 0;
}