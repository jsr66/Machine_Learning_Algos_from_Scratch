ACADEMIC HONESTY
As usual, the standard honour code and academic honesty policy applies. We will be using automated plagiarism detection software to ensure that only original work is given credit. Submissions isomorphic to (1) those that exist anywhere online, (2) those submitted by your classmates, or (3) those submitted by students in prior semesters, will be detected and considered plagiarism.

INSTRUCTIONS
In this assignment you will implement the probabilistic matrix factorization (PMF) model. Recall that this model fills in the values of a missing matrix , where  is an observed value if , where  contains the measured pairs. The goal is to factorize this matrix into a product between vectors such that , where each .

The modeling problem is to learn  for  and  for  by maximizing the objective function


For this problem set ,  and .

Sample starter code to read the inputs and write the outputs:  Download hw4_PMF.py

WHAT YOU NEED TO SUBMIT
You can use either Python or Octave coding languages to complete this assignment. Octave is a free version of Matlab. Your Matlab code should be able to directly run in Octave, but you should not assume that advanced built-in functions will be available to you in Octave. Unfortunately we will not be supporting other languages in this course.

.

Depending on which language you use, we will execute your program using one of the following two commands.

.

Either

$ python hw4_PMF.py ratings.csv

Or

$ octave -q hw4_PMF.m ratings.csv

.

You must name your file as indicated above for your chosen language. If both files are present, we will only run your Python code. We will create and input the csv data file to your code.

.

The csv files that we will input into your code are formatted as follows:.

ratings.csv: A comma separated file containing the data. Each row contains a three values that correspond in order to: user_index, object_index, rating
WHAT YOUR PROGRAM OUTPUTS
You should write your PMF algorithm to learn 5 dimensions. Run your algorithm for 50 iterations.

.

When executed, you will have your code write several output files each described below. It is required that you follow the formatting instructions given below. Where you see [iteration] below, replace this with the iteration number.

.

objective.csv: This is a comma separated file containing the PMF objective function given above along each row. There should be 50 rows and each row should have one value.

.

U-[iteration].csv: This is a comma separated file containing the locations corresponding to the rows, or "users", of the missing matrix . The th row should contain the th user's vector (5 values). You only need to create this file for iteration number 10, 25, and 50. For example, the 10th iteration will produce file U-10.csv

.

V-[iteration].csv: This is a comma separated file containing the locations corresponding to the columns, or "objects",  of the missing matrix . The th row should contain the th object's vector (5 values). You only need to create this file for iteration number 10, 25, and 50. For example, the 10th iteration will produce file V-10.csv

.

Note on Correctness

Please note that for both of these problems, there are multiple potential answers depending on your initialization. However, the PMF algorithm has some known deterministic properties that we discussed in class, and so in this sense we can distinguish between correct and incorrect answers. We strongly suggest that you test out your code on your own computer before submitting.