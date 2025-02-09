# OpenMP Code Overview

This document outlines the codes in the repository focusing on the use of OpenMP for parallel computation.

Return to the [main repository documentation](./README.md).

## 7 - OpenMP Hello World

To build and run - navigate to the directory holding this example and type "make", i.e.:

```bash
cd 7_OMP_Hello_World
make && ./main.exe
```

### Expected Output

The number of "Hello!" messages you will see depends on your machine. Assuming your CPU supports 4 OpenMP threads natively, you'll see this:

```bash
Hello!
Hello!
Hello!
Hello!
```
