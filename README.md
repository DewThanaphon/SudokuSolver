# Sudoku Solver

Author: Thanaphon Rianthong

Contact: dew.thanaphon523@gmail.com

Code Description: This module uses for solving the sudoku problem from an image of a problem and returns a resultant image that was put result digits by a system.

## How to Use

```bash
# in your python code

from SudokuSolver import sudokuSolver

img = 'your/Image/Directory'
answer, img = sudokuSolver(img)

# answer is an array output
# img is an image output in BGR channels
```

**For an example:**

```bash
from SudokuSolver import sudokuSolver

img = 'data/L101.png'
answer, img = sudokuSolver(img)
```

## Example

**image input & image output:**

<img width="357" alt="L302" src="https://user-images.githubusercontent.com/92207106/171144511-a2fda85e-b3e9-4ba5-a3e5-e55259eb88d0.png"> <img width="357" alt="L302Answer" src="https://user-images.githubusercontent.com/92207106/171144589-40f40eea-d84b-4511-a3cd-a96ad7b711ce.png">

**array output:**
```bash
[[4. 3. 9. 6. 2. 5. 1. 7. 8.]
 [6. 1. 7. 3. 9. 8. 4. 2. 5.]
 [8. 5. 2. 4. 1. 7. 6. 3. 9.]
 [3. 6. 5. 2. 8. 1. 9. 4. 7.]
 [1. 7. 8. 9. 6. 4. 2. 5. 3.]
 [9. 2. 4. 7. 5. 3. 8. 1. 6.]
 [2. 8. 1. 5. 7. 6. 3. 9. 4.]
 [7. 9. 3. 8. 4. 2. 5. 6. 1.]
 [5. 4. 6. 1. 3. 9. 7. 8. 2.]]
```

### Errors

If you got an error that was **'OSError: No file or directory found at digitModel.h5'**, you would add one argument in the function **sudokuSolver** to specify a directory of the "digitModel.h5". like this:

```bash
answer, img = sudokuSolver(img, model='your/path/digitModel.h5')
```
