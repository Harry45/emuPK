# Author: Arrykrishna Mootoovaloo
# Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development
# Description : Script ot generate Latin Hypercube Samples with lhs routine 

# Can further specify other methods - see lhs.pdf manual 
# n is the number of points we require 
# d is the dimensionality of the problem 

# import the lhs library
library(lhs)

# number of design points
n = 1000

# number of dimensions
d = 6

# type of design we want to generate
X = maximinLHS(n, d)
# X = optimumLHS(n, d)
# X = randomLHS(n, d)

# save file in the design folder
write.csv(X, 'design/maximin_1000_6D')

