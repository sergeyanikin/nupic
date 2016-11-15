# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------


print """
This program shows how to access the Temporal Memory directly by demonstrating
how to create a TM instance, train it with vectors, get predictions, and
inspect the state.

The code here runs a very simple version of sequence learning, with one
cell per column. The TM is trained with the simple sequence A->B->C->D->E

HOMEWORK: once you have understood exactly what is going on here, try changing
cellsPerColumn to 4. What is the difference between once cell per column and 4
cells per column?

PLEASE READ THROUGH THE CODE COMMENTS - THEY EXPLAIN THE OUTPUT IN DETAIL
"""

# Can't live without numpy
import numpy
from itertools import izip as zip, count

from nupic.research.temporal_memory import TemporalMemory as TM

verbose = True

# Utility routine for printing the input vector
def formatRow(x):
  s = str(x)
  return " ".join(s[i:i+10] for i in range(0, len(s), 10))


def formatList(l):
    return "{} ({})".format(l, len(l))


def formatCells(cells):
    s = "\n"
    bits = idxsToBits(tm.numberOfCells(), cells)
    for r in range(tm.getCellsPerColumn()):
        row = bits[r::tm.getCellsPerColumn()]
        s += formatRow(''.join(row)) + "\n"
    return s


def idxsToBits(width, idxs):
    return ['1' if i in idxs else '0' for i in range(width)]

# Step 1: create Temporal Pooler instance with appropriate parameters

tm = TM(columnDimensions = (50,),
        cellsPerColumn=2,
        initialPermanence=0.5,
        connectedPermanence=0.9,
        minThreshold=4,
        maxNewSynapseCount=20,
        permanenceIncrement=0.05,
        permanenceDecrement=0.0,
        activationThreshold=10,
        )


# Step 2: create input vectors to feed to the temporal memory. Each input vector
# must be numberOfCols wide. Here we create a simple sequence of 5 vectors
# representing the sequence A -> B -> C -> D -> E
x = numpy.zeros((9, tm.numberOfColumns()), dtype="uint32")
x[0, 0:10] = 1    # Input SDR representing "A", corresponding to columns 0-9
x[2, 10:20] = 1   # Input SDR representing "B", corresponding to columns 10-19
x[5, 20:30] = 1   # Input SDR representing "C", corresponding to columns 20-29
x[6, 30:40] = 1   # Input SDR representing "D", corresponding to columns 30-39
x[3, 40:50] = 1   # Input SDR representing "E", corresponding to columns 40-49
x[1, 5:15] = 1    # Input SDR representing "a", corresponding to columns 5-9
x[8, 15:25] = 1   # Input SDR representing "b", corresponding to columns 15-19
x[4, 25:35] = 1   # Input SDR representing "c", corresponding to columns 25-29
x[7, 35:45] = 1   # Input SDR representing "d", corresponding to columns 35-39


# Step 3: send this simple sequence to the temporal memory for learning
# We repeat the sequence 10 times
bored = False
predictedWithoutMistake = 0
predictedWholeSequenceWithoutMistakes = 0
lastPredictedColums = []
step = 0
while not bored:
  print "\n---------- STEP", step+1, "----------"

  # Send each letter in the sequence in order
  for j in range(len(x)):
    activeColumns = set([a for a, b in zip(count(), x[j]) if b == 1])
    if(verbose):
        print "\n\n--------","ABCDEabcde"[j],"-----------"
        print("Input:\t\t" + formatRow(''.join(str(i) for i in x[j])))
        print("Predicted:\t" + formatRow(''.join(str(i) for i in idxsToBits(tm.numberOfColumns(), lastPredictedColums))))

    if(activeColumns == set(lastPredictedColums)):
        predictedWithoutMistake += 1
        print("!!! BINGO :) !!!")
    else:
        predictedWithoutMistake = 0
        print("not expected...")

    # The compute method performs one step of learning and/or inference. Note:
    # here we just perform learning but you can perform prediction/inference and
    # learning in the same step if you want (online learning).
    tm.compute(activeColumns, learn = True)

    lastPredictedColums = [tm.columnForCell(i) for i in tm.getPredictiveCells()]
    # The following print statements can be ignored.
    # Useful for tracing internal states
    if(verbose):
        print("active cells " + formatCells(tm.getActiveCells()))
        print("predictive cells " + formatCells(tm.getPredictiveCells()))
        print("winner cells " + formatCells(tm.getWinnerCells()))
        print("# of segments " + str(tm.connections.numSegments())) + " (" + str(len(tm.getActiveSegments())) + " are active)"
        activeSegments = [s.cell for s in tm.getActiveSegments()]
        matchingSegments = [s.cell for s in tm.getMatchingSegments()]
        print("active cells (by segments) " + formatCells(activeSegments))
        print("matching cells (by segments) " + formatCells(matchingSegments))

  # The reset command tells the TM that a sequence just ended and essentially
  # zeros out all the states. It is not strictly necessary but it's a bit
  # messier without resets, and the TM learns quicker with resets.
  if(predictedWithoutMistake == len(x) - 1):
      predictedWholeSequenceWithoutMistakes += 1
  else:
    predictedWholeSequenceWithoutMistakes = 0
  if(predictedWholeSequenceWithoutMistakes > 3):
    bored = True

  tm.reset()
  step += 1


#######################################################################
#
# Step 3: send the same sequence of vectors and look at predictions made by
# temporal memory
for j in range(len(x)):
  print "\n\n--------","ABCDEabcde"[j],"-----------"
  print "Raw input vector : " + formatRow("".join([str(i) for i in x[j]]))
  activeColumns = set([i for i, j in zip(count(), x[j]) if j == 1])
  # Send each vector to the TM, with learning turned off
  tm.compute(activeColumns, learn = False)

  # The following print statements prints out the active cells, predictive
  # cells, active segments and winner cells.
  #
  # What you should notice is that the columns where active state is 1
  # represent the SDR for the current input pattern and the columns where
  # predicted state is 1 represent the SDR for the next expected pattern
  print "\nAll the active and predicted cells:"
  
  print("active cells " + formatCells(tm.getActiveCells()))
  print("predictive cells " + formatCells(tm.getPredictiveCells()))
  print("winner cells " + formatCells(tm.getWinnerCells()))
  print("# of segments " + str(tm.connections.numSegments())) + " (" + str(len(tm.getActiveSegments())) + " are active)"

  activeColumnsIndeces   = [tm.columnForCell(i) for i in tm.getActiveCells()]
  predictedColumnIndeces = [tm.columnForCell(i) for i in tm.getPredictiveCells()]


  # Reconstructing the active and inactive columns with 1 as active and 0 as
  # inactive representation.

  actColStr  = ("".join(idxsToBits(tm.numberOfColumns(), activeColumnsIndeces)))
  predColStr = ("".join(idxsToBits(tm.numberOfColumns(), predictedColumnIndeces)))

  # For convenience the cells are grouped
  # 10 at a time. When there are multiple cells per column the printout
  # is arranged so the cells in a column are stacked together
  print "Active columns:    " + formatRow(actColStr)
  print "Predicted columns: " + formatRow(predColStr)

print "\nIt took me only", step, "steps to become bored of that excercise"

  # predictedCells[c][i] represents the state of the i'th cell in the c'th
  # column. To see if a column is predicted, we can simply take the OR
  # across all the cells in that column. In numpy we can do this by taking
  # the max along axis 1.
