#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  if (argc != 2) {
    fprintf(stderr, "Missing file name\n");
    exit(1);
  }
  FILE *output = fopen(argv[1], "w+b");
  if (!output) {
    fprintf(stderr, "Unable to open output file\n");
    exit(1);
  } else {
    char buffer[80];
    size_t bytes;
    do {
      bytes = fread(buffer, 1, 80, stdin);
      if (bytes) {
	fwrite(buffer, 1, bytes, stdout);
	fwrite(buffer, 1, bytes, output);
      } else {
	fprintf(stderr, "Error %d\n", errno);
      }
    } while (bytes == 80);
    fclose(output);
    exit(0);
  }
}
