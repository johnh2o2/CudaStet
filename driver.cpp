#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "stetson.h"

/** Maximum character buffer length. */
#define MAXBUF 1000

/** Increment in number of data samples. */
static int INCR = 1000;

/**
 * Sets the increment in data samples
 * for readDataFile().
 *
 * \param newval the new increment value.
 *
 * \return the old increment value.
 */
int setReadDataFileIncrement(int newval)
{
  int old = INCR;
  INCR = newval;
  return old;
}

/**
 * Reads a line and stores it into a buffer.
 *
 * @param line the line to read.
 * @param maxlen the maximal line length.
 * @param fp the \a FILE opened for reading.
 *
 * @return the actual line length (always
 *         less than \a maxlen).
 */
static int readLine(char *line, int maxlen, FILE *fp)
{
  *line = '\0';
  if (fgets(line, maxlen, fp) == NULL)
    return 0;

  return strlen(line);
}


typedef enum { IO_FAILURE, IO_SUCCESS } ioflag;

/**
 * Reallocates memory, or exits the programme
 * in case of an error. 
 *
 * \param p the memory area to be reallocated.
 * \param size the memory space.
 */
void* xrealloc(void* p, size_t size)
{
  void* q = realloc(p, size);
  if (!q){
    fprintf(stderr, "not enough memory for reallocation");
    exit(EXIT_FAILURE);
  }
  return q;
}


void remove_newline(char *s, int max_len){
    for(int i = 0; i < max_len; i++){
        if (s[i] == '\n'){ 
            s[i] = '\0';
            break;
        }
    }
}

/**
 * Reads the datafile.
 *
 * - Input:
 * \param name the name of the file.
 *
 * - Output:
 * \param t points to the sampling times.
 * \param y points to the measurements.
 *
 * \return the nomber of data points read.
 *
 * \note Comment lines (i.e., lines starting with '#')
 *       are skipped.
 * \note The entries of the data tables are dynamically
 *       allocated in chunks of fixed size (default: 1000
 *       entries); this size may be changed with the function
 *       setDataFileIncrement().
 */
ioflag read3colDataFile(const char* name, const int skiprows, real_type** t, real_type** y, real_type** w, int *nlines)
{
  FILE* fp = fopen(name, "r");
  if (!fp) {
    fprintf(stderr, "Cannot open file %s\n", name);
    return IO_FAILURE;
  }

  *t = NULL;
  *y = NULL;
  *w = NULL;
  *nlines = 0;          /* number of data samples found */
  int lineno = 0;

  char buffer[MAXBUF];
  long int len = 0;
  while ((len = readLine(buffer, MAXBUF, fp)) > 0)
  {
    lineno++;
    if (lineno < skiprows + 1) 
      continue;
    
    if (len > MAXBUF - 1)
    {
      fprintf(stderr,
          "line too long (%ld > %d) in file `%s'\n",
          len, MAXBUF, name);
      return IO_FAILURE;
    }

    if ((*nlines) % INCR == 0)
    {
      real_type* tt = (real_type *)xrealloc(*t, 
                        (*nlines + INCR) * sizeof(real_type));
      real_type* yy = (real_type *)xrealloc(*y, 
                        (*nlines + INCR) * sizeof(real_type));
      real_type* ww = (real_type *)xrealloc(*w, 
                        (*nlines + INCR) * sizeof(real_type));
      *t = tt;
      *y = yy;
      *w = ww;
    }
    
    if (*buffer == '#') /* skips line of comments */
      continue;
    
    int retval = sscanf(buffer, "%lf %lf %lf", &(*t)[(*nlines)], 
                                 &(*y)[(*nlines)], &(*w)[(*nlines)]);
    if (retval < 3){
      fprintf(stderr, "incorrect format of input file `%s'\n", name);
      return IO_FAILURE;
    }
    else
      (*nlines)++;
  }
  
  fclose(fp);

  return IO_SUCCESS;
}



ioflag read_list(const char* name, char ***list, int *nlines, const int element_length)
{
  // open and check file
  FILE* fp = fopen(name, "r");
  if (!fp) {
    fprintf(stderr, "Cannot open file %s\n", name);
    return IO_FAILURE;
  }
  
  // initialize
  *nlines = 0;          /* number of data samples found */

  char buffer[MAXBUF];
  long int len = 0;

  // count lines and do error checking
  while ((len = readLine(buffer, MAXBUF, fp)) > 0)
  {
    (*nlines)++;
    
    if (len > element_length - 1)
    {
      fprintf(stderr,
          "line too long (%ld > %d) in file `%s'\n",
          len, MAXBUF, name);
      return IO_FAILURE;
    }
  }

  // allocate memory for list
  *list = (char **) malloc((*nlines) * sizeof(char *));
  for (int i = 0; i < *nlines; i++) 
    (*list)[i] = (char *) malloc(element_length * sizeof(char));

  // rewind
  fseek(fp, 0, SEEK_SET);

  // now read in the list for real
  for (int i = 0; i < *nlines; i++){
    readLine(buffer, element_length, fp);
    remove_newline(buffer, element_length);
    sprintf((*list)[i], "%s", buffer);
  }

  return IO_SUCCESS;
}

void check_io(ioflag fl, const char *task){
    if (fl == IO_FAILURE){
        fprintf(stderr, "%s failed\n", task);
        exit(EXIT_FAILURE);
    }

}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "%s version %s\n", argv[0], VERSION);
    	fprintf(stderr, "usage: %s <filename> <{s, l}> <skiprows>\n\n", argv[0]);
    	
    	fprintf(stderr, "  filename      : path to datafile with either a list of filenames\n"
                        "                  or a path to a single file containing 'x y yerr'\n" 
                        "                  on each line\n");
        fprintf(stderr, "  s             : single file\n");
        fprintf(stderr, "  l             : list of files\n");
    	fprintf(stderr, "  skiprows      : number of rows to skip in the datafile\n");
    	exit(EXIT_FAILURE);
    }

    int max_len = 1000;

    char **filenames;
    int nfiles = 0;

    char file_mode = argv[2][0];

    if (file_mode == 's'){
        filenames = (char **) malloc(sizeof(char *));
        *filenames = (char *) malloc(sizeof(char) * max_len);

        sprintf(filenames[0], "%s", argv[1]);
        nfiles = 1;
    }
    else if (file_mode == 'l')
        check_io(read_list(argv[1], &filenames, &nfiles, max_len), 
             "reading filelist");

    int skiprows = atoi(argv[3]);
    
    
    printf("filename J_exp J_cons\n");
    for(int i = 0; i < nfiles; i++){
        real_type *x, *y, *yerr;
        int N;
        char msg[max_len];
        sprintf(msg, "reading %s", filenames[i]);

        check_io(
            read3colDataFile(filenames[i], skiprows, &x, &y, &yerr, &N),
            msg
        );
        real_type Jexp = stetson_j_gpu(x, y, yerr, EXP, N);
        real_type Jcon = stetson_j_gpu(x, y, yerr, CONSTANT, N);

        free(x); free(y); free(yerr);

        printf("%s %e %e\n", filenames[i], Jexp, Jcon);
    }

    return EXIT_SUCCESS;
}

