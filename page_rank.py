import pandas
import numpy
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import scipy.sparse as sparse

# reading data from csv to data frames
X_df = pandas.read_csv("data/ranking/CFB2016_scores.csv", header=None)

# reading text file
with open("data/ranking/TeamNames.txt", 'r', encoding='utf-8') as f:
    content = f.readlines()
teamNames = [x.strip() for x in content]

# converting data frames to numpy arrays
matches = X_df.as_matrix()


# unnormalized matrix
M = [[0.0 for i in range(760)] for j in range(760)]
M = numpy.matrix(M)

for match in matches:
    teamA = match[0] - 1
    teamB = match[2] - 1
    pointA = match[1]
    pointB = match[3]
    sumOfPts = pointA + pointB
    if(pointA > pointB):
        M[teamA, teamA] += 1 + (pointA / sumOfPts)
        M[teamB, teamA] += 1 + (pointA / sumOfPts)
        M[teamB, teamB] += (pointB / sumOfPts)
        M[teamA, teamB] += (pointB / sumOfPts)
    else:
        M[teamB, teamB] += 1 + (pointB / sumOfPts)
        M[teamA, teamB] += 1 + (pointB / sumOfPts)
        M[teamA, teamA] += (pointA / sumOfPts)
        M[teamB, teamA] += (pointA / sumOfPts)

# normalize rows of M
M = normalize(M, axis=1, norm='l1')


t = [10, 100, 1000, 10000]


w1 = [1 / 760 for i in range(760)]
w1 = numpy.asarray(w1)
w1 = w1.reshape(1, 760)

# lists to hold top 25 teams for each t
top = []
values = []

for value in t:
    wt = numpy.dot(w1, numpy.linalg.matrix_power(M, value))
    value = numpy.sort(wt)
    valueFlip = numpy.fliplr(value)
    ind = numpy.argsort(wt)
    indFlip = numpy.fliplr(ind)
    # pick 25 from ascending order of ind
    top.append(indFlip[:, :25])
    values.append(valueFlip[:, :25])

for teams in top:
    for team in teams:
        for teamNumber in team:
            print(teamNames[teamNumber])


MT = numpy.transpose(M)
firstEigValue, firstEigVector = sparse.linalg.eigs(MT, k=1)
firstEigVector = firstEigVector.real
denom = numpy.sum(firstEigVector)
winf = numpy.divide(firstEigVector.transpose(), denom)

wts = []
wt = w1
for t in range(1, 10001):
    wt = numpy.dot(wt, M)
    wts.append(wt)

winfWt = []
for i in range(10000):
    diff = wts[i] - winf
    winfWt.append(numpy.sum(abs(diff)))

plt.plot(winfWt)