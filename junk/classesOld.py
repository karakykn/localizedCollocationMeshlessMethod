import numpy as np

class Mesh(object):
    """Mesh generator

    Parameters
    -------------
    spatialSteps: 1d-array of integers (size 2)
        dx and dy of the domain.
    vertices: 2d-array
        Vertex points of the polygon (domain).

    Attributes
    -------------
    locations: 2d-array
        x and y locations of the mesh.
    nodeNo: integer
        Total number of points inside the domain.
    faces: 1d-array of 1d arrays
        Indices of the points at the faces.
    interior: 1d array
        Indices of the points inside the domain
    connect: 2d-array
        Connectivity matrix.
    """

    def __init__(self, spatialSteps=np.array([.5,.5]), vertices=np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])):

        """Creates locations, face and interior index and calculates total number of nodes.
        --------------

        Limitations: i)     Vertices must start from the lower left point of the domain.
                     ii)    No vertex must have lower x value from the first vertex.
                     iii)   Vertices must close the polygon counter-clockwise.
        """

        self.spatialSteps = spatialSteps
        self.vertices = vertices

        n = self.vertices.shape[0]
        self.faces = np.array(np.zeros(n-1), dtype=object)
        self.interior = np.array([], dtype=int)


        if (self.vertices[0,:] == self.vertices[n-1,:]).all:
            faceCounter = 0
            for i in range(n-1):
                normal = np.array([self.vertices[i+1,0]-self.vertices[i,0],self.vertices[i+1,1]-self.vertices[i,1]])
                distNormal = np.sqrt(normal[0]**2+normal[1]**2)
                normalHat = normal/distNormal
                counter = 0
                faceIndex = np.array([],dtype = int)
                while True:
                    newPosX = self.vertices[i,0] + self.spatialSteps[0]*normalHat[0]*counter
                    newPosY = self.vertices[i,1] + self.spatialSteps[1]*normalHat[1]*counter
                    distPoint = np.sqrt((newPosX-self.vertices[i,0])**2 + (newPosY-self.vertices[i,1])**2)
                    if distPoint>=distNormal:
                        break
                    if counter == 0 and i == 0:
                        self.locations = np.array([newPosX,newPosY])
                    else:
                        self.locations = np.vstack([self.locations, np.array([newPosX, newPosY])])
                    counter += 1
                    faceIndex = np.append(faceIndex, faceCounter)
                    faceCounter += 1
                self.faces[i] = faceIndex
        else:
            raise Exception("Enter a valid loop!")

        n = vertices.shape[0]
        yChangingVertIndex = np.array([], dtype=int)
        yChangingVertIndexOrie = np.array([], dtype=int)
        for i in range(n - 1):
            if vertices[i + 1, 1] - vertices[i, 1] > 0:
                yChangingVertIndex = np.append(yChangingVertIndex, i)
            elif vertices[i + 1, 1] - vertices[i, 1] < 0:
                yChangingVertIndexOrie = np.append(yChangingVertIndexOrie, i)

        stopper = (np.max(self.vertices[:,1]))
        startingPoint = self.vertices[0, :]
        while True:
            newLocation = np.array([startingPoint[0], startingPoint[1] + self.spatialSteps[1]])
            counter = 0
            while True:
                pointInDomain = pointCheck(newLocation,self.vertices, yChangingVertIndex, yChangingVertIndexOrie)
                if pointInDomain == 1:
                    self.locations = np.vstack([self.locations, newLocation])
                    newLocation = newLocation + np.array([spatialSteps[0],0])
                    self.interior = np.append(self.interior, int(faceCounter))
                    faceCounter += 1
                    counter = 1
                elif counter == 0:
                    newLocation = newLocation + np.array([spatialSteps[0], 0])
                    counter = 1
                else:
                    break
            startingPoint = np.array([startingPoint[0], startingPoint[1] + self.spatialSteps[1]])
            if startingPoint[1] > stopper:
                break

        self.nodeNo = int(self.interior[self.interior.size-1] + 1)

    def connectivity(self):
        """Create connectivity matrix.

        Returns
        -----------
        self.connect: 2d-array
            Connectivity matrix.

        """

        epl = self.intervalNo[0]
        npl=epl+1
        self.connect=np.zeros((2*self.intervalNo[0]*self.intervalNo[1], 3), dtype=int)

        if self.domainShape == 'rectangular':

            for i in range(self.intervalNo[1]):
                for j in range(self.intervalNo[0]):
                    for pp in range(2):
                        if pp % 2 == 0:
                            self.connect[int(2 * i * epl + 2 * j + pp), 0] = (j + 1) + npl * i
                            self.connect[int(2 * i * epl + 2 * j + pp), 1] = (i + 1) * npl + j + 1
                            self.connect[int(2 * i * epl + 2 * j + pp), 2] = j + npl * i
                        if pp % 2 == 1:
                            self.connect[int(2 * i * epl + 2 * j + pp), 0] = j + npl * (i + 1)
                            self.connect[int(2 * i * epl + 2 * j + pp), 1] = j + npl * i
                            self.connect[int(2 * i * epl + 2 * j + pp), 2] = (i + 1) * npl + j + 1
            return self.connect

        else:
            raise Exception("The shape entered for the domain is not defined!")

class LCMM(object):
    """Differential equation solver

    Parameters
    -------------
    mesh: object
        Mesh object which holds mesh properties.
    rbf: string
        Radial basis function.
    diffEq: string
        The type of differential equation to be solved.

    Attributes
    -------------
    f,fx,fy,fxx,fyy: 2d-array
        Radial basis function coefficients.
    system: 2d-array
        System matrix.
    rhs: 1d-array
        Right hand side of the euation, load vector.
    soln: 1d-array
        Solution of the differential equation.

    """

    def __init__(self, mesh, shapeParam, rbf='MQ', boundaries = np.array([0,0,0,0])):

        self.mesh = mesh
        self.rbf = rbf
        self.shapeParam = shapeParam
        self.subDomainRadius = np.max(self.mesh.spatialSteps[0])
        self.boundaries = boundaries

        self.length = self.mesh.nodeNo
        length = self.length
        self.f = np.zeros((length,length))
        self.fx = np.zeros((length,length))
        self.fxx = np.zeros((length,length))
        self.fy = np.zeros((length,length))
        self.fyy = np.zeros((length,length))

        epsilon=1e-12
        self.subDomains = np.array(np.zeros(self.mesh.nodeNo),dtype=object)
        for i in range(self.mesh.nodeNo):
            xi, yi = self.mesh.locations[i,0], self.mesh.locations[i,1]
            influencer = np.array([], dtype=int)
            for j in range(self.mesh.nodeNo):
                xj, yj = self.mesh.locations[j,0], self.mesh.locations[j,1]
                distance = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if distance<=self.subDomainRadius+epsilon:
                    influencer = np.append(influencer, j)
            self.subDomains[i] = influencer

        if self.rbf == 'MQ':

            for i in range(length):
                for j in range(length):
                    self.f[i,j] = np.sqrt( (self.mesh.locations[i,0] - self.mesh.locations[j,0])**2 + (self.mesh.locations[i,1] - self.mesh.locations[j,1])**2 + self.shapeParam**2 )
                    self.fx[i, j] = (self.mesh.locations[i,0] - self.mesh.locations[j,0]) / self.f[i, j]
                    self.fy[i, j] = (self.mesh.locations[i,1] - self.mesh.locations[j,1]) / self.f[i, j]
                    self.fxx[i, j] = 1 / self.f[i, j] - (self.mesh.locations[i,0] - self.mesh.locations[j,0]) ** 2 / self.f[i, j] ** 3
                    self.fyy[i, j] = 1 / self.f[i, j] - (self.mesh.locations[i,1] - self.mesh.locations[j,1]) ** 2 / self.f[i, j] ** 3

        elif self.rbf == 'TPS':
            if self.shapeParam % 2 == 0:
                for i in range(length):
                    for j in range(length):
                        if i != j:
                            r = np.sqrt( (self.mesh.locations[i,0] - self.mesh.locations[j,0])**2 + (self.mesh.locations[i,1] - self.mesh.locations[j,1])**2 )
                            self.f[i,j] = r**self.shapeParam*np.log(r)
                            self.fx[i, j] = (self.mesh.locations[i,0]-self.mesh.locations[j,0])*r**(self.shapeParam-2)*(self.shapeParam*np.log(r)+1)
                            self.fy[i, j] = (self.mesh.locations[i,1]-self.mesh.locations[j,1])*r**(self.shapeParam-2)*(self.shapeParam*np.log(r)+1)
                            self.fxx[i, j] = r**(self.shapeParam-2)*(self.shapeParam*np.log(r)+1)+(self.mesh.locations[i,0]-self.mesh.locations[j,0])**2*r**(self.shapeParam-4)*(2*(self.shapeParam-1)+self.shapeParam*(self.shapeParam-2)*np.log(r))
                            self.fyy[i, j] = r**(self.shapeParam-2)*(self.shapeParam*np.log(r)+1)+(self.mesh.locations[i,1]-self.mesh.locations[j,1])**2*r**(self.shapeParam-4)*(2*(self.shapeParam-1)+self.shapeParam*(self.shapeParam-2)*np.log(r))
            else:
                raise Exception('Enter a positive even number as a shape parameter for TPS.')
        else:
            raise Exception('The radial basis function entered is not defined.')

        self.localRBF = np.zeros((self.length,self.length))
        self.system = np.zeros((self.length,self.length))
        self.rhs = np.zeros(self.length)

        for k in range(self.length):
            self.localRBF[k,self.subDomains[k][:]] = self.f[k,self.subDomains[k][:]]

    def steadyPoisson(self, materialCoefficient=1):
        for k in range(self.length):
            key = onWhichFace(self.mesh.faces, self.boundaries, k)
            if key[0] == 0:
                self.system[k, self.subDomains[k][:]] = self.f[k, self.subDomains[k][:]]
                self.rhs[k] = unknown(self.mesh.locations[k,0], self.mesh.locations[k,1])
            elif key[0] == 1:
                faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
                self.system[k, self.subDomains[k][:]] = faceNormal[0]*self.fx[k, self.subDomains[k][:]] + faceNormal[1]*self.fy[k, self.subDomains[k][:]]
                self.rhs[k] = du_dn(self.mesh.locations[k,0], self.mesh.locations[k,1])
            else:
                self.system[k, self.subDomains[k][:]] = -materialCoefficient*(self.fxx[k, self.subDomains[k][:]] + self.fyy[k, self.subDomains[k][:]])
                self.rhs[k] = source(self.mesh.locations[k, 0], self.mesh.locations[k, 1])

    def steadySolve(self):
        invSystem=np.linalg.pinv(self.system)
        self.alpha = np.matmul(invSystem, self.rhs)
        self.soln = np.matmul(self.localRBF,self.alpha)

    def unsteadyPoisson(self, materialCoefficient=1, timeStep=1e-1):
        self.materialCoefficient = materialCoefficient
        self.timeStep = timeStep
        for k in range(self.length):
            key = onWhichFace(self.mesh.faces, self.boundaries, k)
            if key[0] == 0:
                self.system[k, self.subDomains[k][:]] = self.f[k, self.subDomains[k][:]]
            elif key[0] == 1:
                faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
                self.system[k, self.subDomains[k][:]] = faceNormal[0] * self.fx[k, self.subDomains[k][:]] + faceNormal[
                    1] * self.fy[k, self.subDomains[k][:]]
            else:
                self.system[k, self.subDomains[k][:]] = self.localRBF[k, self.subDomains[k][:]] - timeStep * materialCoefficient * (self.fxx[k, self.subDomains[k][:]] + self.fyy[k, self.subDomains[k][:]])

    def unsteadyPoissonRHS(self, time):
        for k in range(self.length):
            key = onWhichFace(self.mesh.faces, self.boundaries, k)
            if key[0] == 0:
                self.rhs[k] = unsteadyUnknown(self.mesh.locations[k, 0], self.mesh.locations[k, 1], (time+1)*self.timeStep)
            elif key[0] == 1:
                self.rhs[k] = unsteady_du_dn(self.mesh.locations[k, 0], self.mesh.locations[k, 1], (time+1)*self.timeStep)
            else:
                self.rhs[k] = self.timeStep * unsteadySource(self.mesh.locations[k, 0], self.mesh.locations[k, 1], (time+1)*self.timeStep)

    def unsteadyPoissonSolve(self, caseTime=10):
        timeIter = int(caseTime / self.timeStep)
        invSystem = np.linalg.pinv(self.system)
        self.soln = np.zeros(self.length)
        for i in range(self.length):
            self.soln[i] = unsteadyUnknown(self.mesh.locations[i,0], self.mesh.locations[i,1], 0)
        # Open the two lines below to enter initial solution field, and comment out two lines above.
        # for i in range(self.mesh.faces.size):
        #     self.soln[self.mesh.faces[i][:]] = 1
        for time in range(timeIter):
            self.unsteadyPoissonRHS(time)
            for i in self.mesh.interior:
                self.rhs[i] += self.soln[i]
            self.alpha = np.matmul(invSystem, self.rhs)
            self.soln = np.matmul(self.localRBF, self.alpha)

"""Problem definition"""
def source(x,y):
    return 0
def unknown(x,y):
    return 1
def du_dn(x,y):
    return 0

def unsteadySource(x,y,t):
    return 0
def unsteadyUnknown(x,y,t):
    return 1
def unsteady_du_dn(x,y,t):
    return 0
"""------------------"""

"""Seperate functions"""
def onWhichFace(faces,boundaries,pointIndex):
    n = int(len(faces))
    for i in range(n):
        if pointIndex in faces[i]:
            return np.array([boundaries[i],i])
    return np.array([2,-1])

def faceNormalNeumann(vertices, vertexIndex):
    x1, y1 = vertices[vertexIndex,0], vertices[vertexIndex,1]
    x2, y2 = vertices[vertexIndex+1,0], vertices[vertexIndex+1,1]
    return np.array([(y2-y1),(x1-x2)]) / np.sqrt((x2-x1)**2 + (y2-y1)**2)

def onSegment(p, q, r):
    if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False

def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):

        # Clockwise orientation
        return 1
    elif (val < 0):

        # Counterclockwise orientation
        return 2
    else:

        # Colinear orientation
        return 0

def doIntersect(p1, q1, p2, q2):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # Special Cases

    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return False

    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return False

    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return False

    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return False

    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True

    # If none of the cases
    return False

def pointCheck(point, vertices, yChangingVertIndex, yChangingVertIndexOrie):
    epsilon = 1e-12
    dummyPoint = point + np.array([0,epsilon])
    intersectionCounter = 0
    for i in yChangingVertIndex:
        farPoint = np.array([np.max(vertices[yChangingVertIndex, 0]) + 10, dummyPoint[1]])
        if doIntersect(dummyPoint, farPoint, vertices[i,:], vertices[i+1,:]):
            intersectionCounter +=1
    for i in yChangingVertIndexOrie:
        farPoint = np.array([np.min(vertices[yChangingVertIndex, 0]) - 10, dummyPoint[1]])
        if doIntersect(dummyPoint, farPoint, vertices[i,:], vertices[i+1,:]):
            intersectionCounter +=1
    if intersectionCounter == 2:
        inOrNot = 1
    else:
        inOrNot = 0
    return inOrNot
"""-------------------"""

"""Junk functions"""
# def poisLocals(self, k):
#     n = len(self.subDomains[k])
#     localCol = np.zeros((n, n))
#     for i in range(n):
#         key = onWhichFace(self.mesh.faces, self.boundaries, self.subDomains[k][i])
#         if key[0] == 0:
#             for j in range(n):
#                 localCol[i, j] = self.f[self.subDomains[k][i], self.subDomains[k][j]]
#         elif key[0] == 1:
#             for j in range(n):
#                 faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
#                 localCol[i, j] = faceNormal[0] * self.fx[self.subDomains[k][i], self.subDomains[k][j]] + faceNormal[1] * \
#                                  self.fyy[self.subDomains[k][i], self.subDomains[k][j]]
#         else:
#             for j in range(n):
#                 localCol[i, j] = -(self.fxx[self.subDomains[k][i], self.subDomains[k][j]] + self.fyy[
#                     self.subDomains[k][i], self.subDomains[k][j]])
#     return localCol
# def poisSourceLocals(self, k):
#     n = len(self.subDomains[k])
#     localSources = np.zeros(n)
#     for i in range(n):
#         key = onWhichFace(self.mesh.faces, self.boundaries, self.subDomains[k][i])
#         x = self.mesh.locations[self.subDomains[k][i], 0]
#         y = self.mesh.locations[self.subDomains[k][i], 1]
#         if key[0] == 0:
#             localSources[i] = unknown(x, y)
#         elif key[0] == 1:
#             localSources[i] = du_dn(x, y)
#         else:
#             localSources[i] = source(x, y)
#     return localSources
# for k in range(self.length):
#     localCol = self.poisLocals(k)
#     n = localCol.shape[0]
#     for i in range(n):
#         for j in range(n):
#             self.system[self.subDomains[k][i],self.subDomains[k][j]] += localCol[i,j]
#
# for k in range(self.length):
#     localSou = self.poisSourceLocals(k)
#     n = localSou.size
#     for i in range(n):
#         self.rhs[self.subDomains[k][i]] += localSou[i]
"""--------------"""