#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "stetson.h"
#include "stetson_mean.h"
#include "utils.h"
/** Maximum character buffer length. */
#define MAXBUF 1000

/** Increment in number of data samples. */
static int INCR = 1000;

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
    fprintf(stderr, "not enough memory for reallocation\n");
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
  char fmt[12];
  if (sizeof(real_type) == sizeof(double)) 
      sprintf(fmt, "%%lf %%lf %%lf");
  else
      sprintf(fmt, "%%f %%f %%f");

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
    
    
    int retval = sscanf(buffer, fmt, &(*t)[(*nlines)], &(*y)[(*nlines)], 
                                                       &(*w)[(*nlines)]);
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
  fclose(fp);

  return IO_SUCCESS;
}

void check_io(ioflag fl, const char *task){
    if (fl == IO_FAILURE){
        fprintf(stderr, "%s failed\n", task);
        exit(EXIT_FAILURE);
    }

}

void
run_batch( char **filenames, const int nfiles, const int skiprows, const int batch_size,
            const size_t memlim, real_type **Jexp, real_type **Jcons ){

    real_type *x, *y, *yerr;
    real_type *xn, *yn, *ynerr;
    int *N = (int *)malloc(batch_size * sizeof(int));

    *Jcons = (real_type *) malloc(nfiles * sizeof(real_type));
    *Jexp  = (real_type *) malloc(nfiles * sizeof(real_type));
    real_type *Jtemp;

    int bsize = 0, npoints = 0;
    char msg[MAXBUF];
    size_t mem = 0;
    clock_t start;
    double dt;
    for(int i = 0; i < nfiles; i++){
        //printf("reading file %d (%s)\n", i+1, filenames[i]); 
        //fflush(stdout);
        sprintf(msg, "reading file %d failed\n", i+1);
        // read file
        //start = clock();
        check_io(read3colDataFile(filenames[i], skiprows, &xn, &yn, 
               &ynerr, N + bsize), msg);
        //dt = ((double) (clock() - start))/CLOCKS_PER_SEC;

        //printf("  %e seconds to read file %s\n", dt, filenames[i]);

        if(N[bsize] == 0) {
            fprintf(stderr, "No points read in from %s\n", filenames[i]);
            exit(EXIT_FAILURE);
        }

        npoints += N[bsize];
        
        mem = npoints * (4 * sizeof(real_type) + sizeof(int));
        
        // add more memory
        if (npoints == N[bsize]) {
            x = (real_type *)malloc( npoints * sizeof(real_type));
            y = (real_type *)malloc( npoints * sizeof(real_type));
            yerr = (real_type *)malloc( npoints * sizeof(real_type));
        } else {
            x = (real_type *)xrealloc(x, npoints * sizeof(real_type));
            y = (real_type *)xrealloc(y, npoints * sizeof(real_type));
            yerr = (real_type *)xrealloc(yerr, npoints * sizeof(real_type));
        }

        // copy new data to array
        memcpy(x + (npoints - N[bsize]), xn, N[bsize] * sizeof(real_type));
        memcpy(y + (npoints - N[bsize]), yn, N[bsize] * sizeof(real_type));
        memcpy(yerr + (npoints - N[bsize]), ynerr, N[bsize] * sizeof(real_type));

        bsize += 1;

        // send off batch
        if ( (bsize % batch_size == 0) || (i == nfiles - 1) || mem >= memlim ){
            //printf("sending off batch of size %d\n", bsize);

            //printf("constant\n"); fflush(stdout);
            //start = clock();
            // CONSTANT weighting
            Jtemp = stetson_j_gpu_batch(x, y, yerr, CONSTANT, N, bsize);
            //dt = ((double) (clock() - start))/CLOCKS_PER_SEC;

            //printf("done. Took %e seconds for %d files = %e/file\n", dt, bsize, dt / ((real_type) bsize));
            memcpy(*Jcons + (i - bsize + 1), Jtemp, bsize * sizeof(real_type)); 
            free(Jtemp);

            //printf("exponential\n"); fflush(stdout);
            // EXPONENTIAL weighting
            //start = clock();
            Jtemp = stetson_j_gpu_batch(x, y, yerr, EXP, N, bsize);
            //dt = ((double) (clock() - start))/CLOCKS_PER_SEC;

            //printf("done. Took %e seconds for %d files = %e/file\n", dt, bsize, dt / ((real_type) bsize));
            memcpy(*Jexp + (i - bsize + 1), Jtemp, bsize * sizeof(real_type)); 
            free(Jtemp);

            bsize = 0;
            npoints = 0;
            mem = 0;
            free(x); free(y); free(yerr);
        }

        // free temporary pointers
        free(xn); free(yn); free(ynerr);
    }

    free(N);

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
    
    
    char msg[max_len];
    real_type *Jexp_b, *Jcon_b;
    clock_t  start;
    double dt;

    int batch_size = 10000;
    real_type memlim = 3 * 10E8;
    //start = clock();
    run_batch(filenames, nfiles, skiprows, batch_size, memlim, &Jexp_b, &Jcon_b);
    //dt = ((double) (clock() - start))/CLOCKS_PER_SEC;

    // printf("processed %d files in %e seconds (%e s / file)\n", nfiles, dt, dt/nfiles);

    printf("filename J_exp J_cons K ymean ymean_stet\n");
    for(int i = 0; i < nfiles; i++){
        real_type *x, *y, *yerr;
        int N;
        sprintf(msg, "reading %s", filenames[i]);

        check_io(
            read3colDataFile(filenames[i], skiprows, &x, &y, &yerr, &N),
            msg
        );

	   real_type ymean = mean(y,N);
	   real_type ystetmean = stetson_mean(y, yerr, APARAM, BPARAM, CRITERION, N);
	   real_type K    = stetson_k(y, yerr, N);
       //real_type Jexp = stetson_j_gpu(x, y, yerr, EXP, N);
       //real_type Jcon = stetson_j_gpu(x, y, yerr, CONSTANT, N);

        free(x); free(y); free(yerr);

        printf("%s %e %e %e %e %e\n", filenames[i], Jexp_b[i], Jcon_b[i], K, ymean, ystetmean);
    }

    for(int i = 0; i < nfiles; i++) 
       free(filenames[i]);
    free(filenames);

    return EXIT_SUCCESS;
}

