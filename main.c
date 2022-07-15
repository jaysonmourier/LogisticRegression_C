/**
 * 
 * Logistic regression
 * Jayson Mourier 2022
 * 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <string.h>

#define BUFFER_SZ 4096

FILE* load_data_from_file(const char * path)
{
    FILE* tmp = fopen(path, "r+");

    if(NULL == tmp)
    {
        /*
            ERROR, check errno
        */

       fprintf(stderr, "[ERROR:load_data_from_file] errno say: %s\n", strerror(errno));
       exit(EXIT_FAILURE);
    }

    return tmp;
}

int fill_data(FILE* f, double* arr, const int rows, const int cols)
{
    char buffer[BUFFER_SZ];
    
    const char * separators = ",";

    int count_rows = 0;
    int count_cols = 0;

    while(!feof(f))
    {
        fgets(buffer, BUFFER_SZ, f);

        if(ferror(f))
        {
            fprintf(stderr, "[ERROR:fill_data] errno say: %s\n", strerror(errno));
            return -1;
        }

        const char * strToken = strtok(buffer, separators);

        if(count_rows > rows)
        {
            printf("[WARNING] count_rows > rows!\n");
        }

        while(strToken != NULL)
        {
            //arr[count_cols + count_rows * cols] = (double) atof(strToken);
            if(sscanf(strToken, "%lf", &arr[count_cols + count_rows * cols]) < 1) 
            { 
                return -1;
            }

            strToken = strtok(NULL, separators);
            ++count_cols;
        }

        ++count_rows;
        count_cols = 0;
    }

    return 0;
}

double* create_array(const int rows, const int cols)
{
    if((rows < 1) || (cols < 1))
    {
        fprintf(stderr, "[ERROR:create_array] the dims can't be equal to zero or be negative.\n");
        exit(EXIT_FAILURE);
    }

    double* tmp = (double*) calloc(rows * cols, sizeof(double));

    if(NULL == tmp)
    {
        fprintf(stderr, "[ERROR:create_array] errno say: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    return tmp;
}

void print_array(const double* arr, const int rows, const int cols)
{
    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            printf("%lf, ", arr[j + i * cols]);
        }
        printf("\n");
    }
}

double logloss(double y, double p)
{
    return - y * log(p) - (1 - y) * log(1 - p);
}

double model(double t)
{
    return 1 / (1 + exp(-(t)));
}

double* train(double* data, const int rows, const int cols, const int epochs, const double lr)
{
    double* weigths = (double*) calloc(cols, sizeof(double));
    
    if(NULL == weigths)
    {
        fprintf(stderr, "[ERROR:train] errno say: %s\n", strerror(errno));
        return NULL;
    }

    double total_loss = .0;

    for(int epoch=0; epoch < epochs; ++epoch)
    {

        total_loss = .0;

        for(int i = 0; i < rows; ++i)
        {

            double y = data[cols - 1 + i * cols];
            double t = 0.;
            double x_memory[cols - 1];

            for(int j = 0; j < cols - 1; ++j)
            {
                t += weigths[j] * data[j + i * cols];
                x_memory[j] = data[j + i * cols];
            }

            // add the bias
            t += weigths[cols];

            // get the probability            
            double p = model(t);

            // fit weights
            for(int k = 0; k < cols; ++k)
            {
                weigths[k] = weigths[k] - lr * ( (p - y) * x_memory[k] );
            }

            total_loss += logloss(y, p);
        }

        printf("Loss: %f\n", (total_loss / (double) epoch));

    }

    return weigths;
}

int
main(int argc, char** argv)
{
    --argc, ++argv;
    
    if(argc < 4)
    {
        fprintf(stderr, "[ERROR] Usage: ./program data.txt n_rows n_cols learning_rate\n");
        exit(EXIT_FAILURE);
    }

    char data_path[255];
    int rows;
    int cols;
    double lr;

    if(sscanf(argv[0], "%255s", data_path) < 1) fprintf(stderr, "[ERROR] Usage: ./program data.txt[str] n_rows[int] n_cols[int]"), exit(EXIT_FAILURE);
    if(sscanf(argv[1], "%d", &rows) < 1) fprintf(stderr, "[ERROR] Usage: ./program data.txt[str] n_rows[int] n_cols[int] learning_rate[double]"), exit(EXIT_FAILURE);
    if(sscanf(argv[2], "%d", &cols) < 1) fprintf(stderr, "[ERROR] Usage: ./program data.txt[str] n_rows[int] n_cols[int] learning_rate[double]"), exit(EXIT_FAILURE);
    if(sscanf(argv[3], "%lf", &lr) < 1) fprintf(stderr, "[ERROR] Usage: ./program data.txt[str] n_rows[int] n_cols[int] learning_rate[double]"), exit(EXIT_FAILURE);
    if (rows < 1 || cols < 2 || lr <= 0.) fprintf(stderr, "[ERROR] Usage: ./program data.txt[str<255] n_rows[int>0] n_cols[int>2] learning_rate[double>0]"), exit(EXIT_FAILURE);

    FILE* f = load_data_from_file(data_path);
    
    double* data = create_array(rows, cols);

    if(0 != fill_data(f, data, rows, cols))
    {
        fprintf(stderr, "[ERROR] Unable to convert data from file to double.\n");
        free(data);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    double* w = train(data, rows, cols, 2000, lr);

    if (NULL == w) goto exit;

    print_array(w, 1, cols);

    // cleaning
    free(w);

exit:
    free(data);
    fclose(f);
    
    printf("End of the program.\n");
 
    return 0;
}