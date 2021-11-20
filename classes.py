import numpy as np
import pandas as pd
import time as tm

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

        self.interfaceBNN = 0

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
        self.boundaryNodeNo = 0
        for i in range(self.faces.size):
            self.boundaryNodeNo += self.faces[i].shape[0]

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

    def interface(self, r, shape='deltoid', intBoundary_dtheta=np.pi/8):
        self.shape=shape
        if shape=='deltoid':
            theta = np.arange(0,2*np.pi,intBoundary_dtheta)
            self.theta = theta
            self.interfaceLocs = np.zeros((theta.size,2))
            self.interfaceLocs[:,0] = r*(np.cos(theta))**3
            self.interfaceLocs[:,1] = r*(np.sin(theta))**3
            self.locations = np.append(self.locations, self.interfaceLocs, axis=0)
            self.interfaceBNN = self.interfaceLocs.shape[0]
            self.interfaceInterior = 0
            self.r = r
            i = 0
            while i<self.nodeNo:
                if insideDeltoid(self.locations[i,0],self.locations[i,1],r):
                    self.interfaceLocs = np.append(self.interfaceLocs, [np.array([self.locations[i,0],self.locations[i,1]])], axis=0)
                    self.interfaceInterior += 1
                    ind = np.where(self.interior==i)
                    self.interior = np.delete(self.interior, ind, 0)
                    self.locations = np.delete(self.locations,i, 0)
                    self.nodeNo -= 1
                    for j in self.interior:
                        if j>i:
                            self.interior[np.where(self.interior==j)] -= 1
                else:
                    i += 1
            # self.nodeNo += self.interfaceBNN
            # self.boundaryNodeNo += self.interfaceBNN
        elif shape=='cercle':
            theta = np.arange(0,2*np.pi,intBoundary_dtheta)
            self.theta = theta
            self.interfaceLocs = np.zeros((theta.size,2))
            self.interfaceLocs[:,0] = r*(np.cos(theta))
            self.interfaceLocs[:,1] = r*(np.sin(theta))
            self.locations = np.append(self.locations, self.interfaceLocs, axis=0)
            self.interfaceBNN = self.interfaceLocs.shape[0]
            self.interfaceInterior = 0
            self.r = r
            i = 0
            while i<self.nodeNo:
                if insideCercle(self.locations[i,0],self.locations[i,1],r):
                    self.interfaceLocs = np.append(self.interfaceLocs, [np.array([self.locations[i,0],self.locations[i,1]])], axis=0)
                    self.interfaceInterior += 1
                    ind = np.where(self.interior==i)
                    self.interior = np.delete(self.interior, ind, 0)
                    self.locations = np.delete(self.locations,i, 0)
                    self.nodeNo -= 1
                    for j in self.interior:
                        if j>i:
                            self.interior[np.where(self.interior==j)] -= 1
                else:
                    i += 1

        elif shape=='astroid': ### Not corrected
            theta = np.arange(0,2*np.pi,intBoundary_dtheta)
            self.theta = theta
            self.interfaceLocs = np.zeros((theta.size,2))
            self.interfaceLocs[:,0] = r*(np.cos(theta))
            self.interfaceLocs[:,1] = r*(np.sin(theta))
            self.locations = np.append(self.locations, self.interfaceLocs, axis=0)
            self.interfaceBNN = self.interfaceLocs.shape[0]
            self.interfaceInterior = 0
            self.r = r
            i = 0
            while i<self.nodeNo:
                if insideCercle(self.locations[i,0],self.locations[i,1],r):
                    self.interfaceLocs = np.append(self.interfaceLocs, [np.array([self.locations[i,0],self.locations[i,1]])], axis=0)
                    self.interfaceInterior += 1
                    ind = np.where(self.interior==i)
                    self.interior = np.delete(self.interior, ind, 0)
                    self.locations = np.delete(self.locations,i, 0)
                    self.nodeNo -= 1
                    for j in self.interior:
                        if j>i:
                            self.interior[np.where(self.interior==j)] -= 1
                else:
                    i += 1

class leeMQ(object):
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
        self.subDomainRadius = self.mesh.spatialSteps[0]
        self.boundaries = boundaries

        boundaryNodeNo = self.mesh.boundaryNodeNo
        subDomNo = self.mesh.nodeNo - boundaryNodeNo
        self.subDomNo = subDomNo
        epsilon=1e-14
        self.subDomains = np.array(np.zeros(subDomNo),dtype=object)
        jj = 0
        for i in range(boundaryNodeNo,self.mesh.nodeNo):
            xi, yi = self.mesh.locations[i,0], self.mesh.locations[i,1]
            influencer = np.array([], dtype=int)
            for j in range(self.mesh.nodeNo):
                xj, yj = self.mesh.locations[j,0], self.mesh.locations[j,1]
                distance = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if distance<=self.subDomainRadius+epsilon:
                    influencer = np.append(influencer, j)
            self.subDomains[jj] = influencer
            jj += 1

        dummy = np.array([])
        for i in range(subDomNo):
            dummy = np.concatenate([dummy, self.subDomains[i]])
        holder = getMissing(self.mesh.nodeNo, dummy)
        for k in range(subDomNo):
            for i in range(self.subDomains[k].size):
                checker = self.subDomains[k][i]
                for ii in holder:
                    if checker>ii:
                        self.subDomains[k][i] = self.subDomains[k][i] - 1
        for i in range(self.mesh.interior.size):
            checker = self.mesh.interior[i]
            for ii in holder:
                if checker>ii:
                    self.mesh.interior[i] = self.mesh.interior[i] - 1
        for k in range(self.mesh.faces.size):
            j = 0
            for i in self.mesh.faces[k]:
                for ii in holder:
                    if i == ii:
                        self.mesh.faces[k] = np.delete(self.mesh.faces[k], j, 0)
                j += 1
        for k in range(self.mesh.faces.size):
            j = 0
            for i in self.mesh.faces[k]:
                for ii in holder:
                    if i > ii:
                        self.mesh.faces[k][j] = self.mesh.faces[k][j] - 1
                j += 1
        holder_mod = holder
        for i in range(holder.size):
            holder_mod[i] = holder[i] - i
        for i in holder_mod:
            self.mesh.locations = np.delete(self.mesh.locations, i, 0)
            self.mesh.nodeNo = self.mesh.nodeNo - 1
            self.mesh.boundaryNodeNo = self.mesh.boundaryNodeNo - 1



        self.f = np.array(np.zeros(subDomNo),dtype=object)
        self.fx = np.array(np.zeros(subDomNo),dtype=object)
        self.fxx = np.array(np.zeros(subDomNo),dtype=object)
        self.fy = np.array(np.zeros(subDomNo),dtype=object)
        self.fyy = np.array(np.zeros(subDomNo),dtype=object)
        self.invPnns = np.array(np.zeros(subDomNo), dtype=object)

        if self.rbf == 'MQ':

            for k in range(subDomNo):
                subDomNodeNo = self.subDomains[k].shape[0]
                phiHat = np.zeros((subDomNodeNo,subDomNodeNo))
                phiHat_x = np.zeros((subDomNodeNo,subDomNodeNo))
                phiHat_y = np.zeros((subDomNodeNo,subDomNodeNo))
                phiHat_xx = np.zeros((subDomNodeNo,subDomNodeNo))
                phiHat_yy = np.zeros((subDomNodeNo,subDomNodeNo))
                for i in range(subDomNodeNo):
                    for j in range(subDomNodeNo):
                        r_sq = (self.mesh.locations[self.subDomains[k][i],0] - self.mesh.locations[self.subDomains[k][j],0])**2 + (self.mesh.locations[self.subDomains[k][i],1] - self.mesh.locations[self.subDomains[k][j],1])**2
                        phiHat[i,j] = np.sqrt( r_sq + self.shapeParam**2 )
                        phiHat_x[i, j] = (self.mesh.locations[self.subDomains[k][i],0] - self.mesh.locations[self.subDomains[k][j],0]) / phiHat[i, j]
                        phiHat_y[i, j] = (self.mesh.locations[self.subDomains[k][i],1] - self.mesh.locations[self.subDomains[k][j],1]) / phiHat[i, j]
                        phiHat_xx[i, j] = 1 / phiHat[i, j] - (self.mesh.locations[self.subDomains[k][i],0] - self.mesh.locations[self.subDomains[k][j],0]) ** 2 / phiHat[i, j] ** 3
                        phiHat_yy[i, j] = 1 / phiHat[i, j] - (self.mesh.locations[self.subDomains[k][i],1] - self.mesh.locations[self.subDomains[k][j],1]) ** 2 / phiHat[i, j] ** 3
                self.f[k] = phiHat
                self.fx[k] = phiHat_x
                self.fy[k] = phiHat_y
                self.fxx[k] = phiHat_xx
                self.fyy[k] = phiHat_yy
                self.invPnns[k] = np.linalg.pinv(self.f[k])

        elif self.rbf == 'TPS':
            if self.shapeParam % 2 == 0:
                for k in range(subDomNo):
                    subDomNodeNo = self.subDomains[k].shape[0]
                    phiHat = np.zeros((subDomNodeNo, subDomNodeNo))
                    phiHat_x = np.zeros((subDomNodeNo, subDomNodeNo))
                    phiHat_y = np.zeros((subDomNodeNo, subDomNodeNo))
                    phiHat_xx = np.zeros((subDomNodeNo, subDomNodeNo))
                    phiHat_yy = np.zeros((subDomNodeNo, subDomNodeNo))
                    for i in range(subDomNodeNo):
                        for j in range(subDomNodeNo):
                            if i != j:
                                r = np.sqrt( (self.mesh.locations[self.subDomains[k][i],0] - self.mesh.locations[self.subDomains[k][j],0])**2 + (self.mesh.locations[self.subDomains[k][i],1] - self.mesh.locations[self.subDomains[k][j],1])**2 )
                                phiHat[i,j] = r**self.shapeParam*np.log(r)
                                phiHat_x[i, j] = (self.mesh.locations[self.subDomains[k][i],0]-self.mesh.locations[self.subDomains[k][j],0])*r**(self.shapeParam-2)*(self.shapeParam*np.log(r)+1)
                                phiHat_y[i, j] = (self.mesh.locations[self.subDomains[k][i],1]-self.mesh.locations[self.subDomains[k][j],1])*r**(self.shapeParam-2)*(self.shapeParam*np.log(r)+1)
                                phiHat_xx[i, j] = r**(self.shapeParam-2)*(self.shapeParam*np.log(r)+1)+(self.mesh.locations[self.subDomains[k][i],0]-self.mesh.locations[self.subDomains[k][j],0])**2*r**(self.shapeParam-4)*(2*(self.shapeParam-1)+self.shapeParam*(self.shapeParam-2)*np.log(r))
                                phiHat_yy[i, j] = r**(self.shapeParam-2)*(self.shapeParam*np.log(r)+1)+(self.mesh.locations[self.subDomains[k][i],1]-self.mesh.locations[self.subDomains[k][j],1])**2*r**(self.shapeParam-4)*(2*(self.shapeParam-1)+self.shapeParam*(self.shapeParam-2)*np.log(r))
                    self.f[k] = phiHat
                    self.fx[k] = phiHat_x
                    self.fy[k] = phiHat_y
                    self.fxx[k] = phiHat_xx
                    self.fyy[k] = phiHat_yy
                    self.invPnns[k] = np.linalg.pinv(self.f[k])
            else:
                raise Exception('Enter a positive even number as a shape parameter for TPS.')
        else:
            raise Exception('The radial basis function entered is not defined.')


        self.system = np.zeros((self.mesh.nodeNo, self.mesh.nodeNo))
        self.rhs = np.zeros(self.mesh.nodeNo)

        self.dirichletNodes = np.array([], dtype=int)
        self.neumannNodes = np.array([], dtype=int)
        self.robinNodes = np.array([], dtype=int)

    def steadyPoisson(self, materialCoefficient=1):
        for k in range(self.subDomNo):
            for i in range(self.subDomains[k].shape[0]):
                key = onWhichFace(self.mesh.faces, self.boundaries, self.subDomains[k][i])
                if key[0] == 0:
                    self.system[self.subDomains[k][i], self.subDomains[k][:]] += np.matmul(self.f[k][i,:],self.invPnns[k])
                    self.rhs[self.subDomains[k][i]] += unknown(self.mesh.locations[self.subDomains[k][i],0], self.mesh.locations[self.subDomains[k][i],1])
                    if self.subDomains[k][i] not in self.dirichletNodes:
                        self.dirichletNodes = np.append(self.dirichletNodes, self.subDomains[k][i])
                elif key[0] == 1:
                    faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
                    self.system[self.subDomains[k][i], self.subDomains[k][:]] += np.matmul(faceNormal[0]*self.fx[k][i,:] + faceNormal[1]*self.fy[k][i,:],self.invPnns[k])
                    self.rhs[self.subDomains[k][i]] += du_dn(self.mesh.locations[self.subDomains[k][i],0], self.mesh.locations[self.subDomains[k][i],1])
                    if self.subDomains[k][i] not in self.neumannNodes:
                        self.neumannNodes = np.append(self.neumannNodes, self.subDomains[k][i])
                elif key[0] == 2:
                    faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
                    robinCoefs = robinCoefficients()
                    self.system[self.subDomains[k][i], self.subDomains[k][:]] += np.matmul(robinCoefs[0]*self.f[k][i,:] + robinCoefs[1]*(faceNormal[0]*self.fx[k][i,:] + faceNormal[1]*self.fy[k][i,:]),self.invPnns[k])
                    self.rhs[self.subDomains[k][i]] += robin(self.mesh.locations[self.subDomains[k][i],0], self.mesh.locations[self.subDomains[k][i],1])
                    if self.subDomains[k][i] not in self.robinNodes:
                        self.robinNodes = np.append(self.robinNodes, self.subDomains[k][i])
                else:
                    self.system[self.subDomains[k][i], self.subDomains[k][:]] += np.matmul(-materialCoefficient*(self.fxx[k][i,:] + self.fyy[k][i,:]),self.invPnns[k])
                    self.rhs[self.subDomains[k][i]] += source(self.mesh.locations[self.subDomains[k][i], 0], self.mesh.locations[self.subDomains[k][i], 1])

    def steadySolve(self):
        invSystem=np.linalg.pinv(self.system)
        self.soln = np.matmul(invSystem,self.rhs)

    def unsteadyPoisson(self, dt, timeScheme='backwardEuler', materialCoefficient=1):
        self.dt = dt
        self.timeScheme = timeScheme

        if timeScheme == 'backwardEuler':
            for k in range(self.subDomNo):
                for i in range(self.subDomains[k].shape[0]):
                    key = onWhichFace(self.mesh.faces, self.boundaries, self.subDomains[k][i])
                    if key[0] == 0:
                        self.system[self.subDomains[k][i], self.subDomains[k][:]] += np.matmul(self.f[k][i,:],self.invPnns[k])
                        if self.subDomains[k][i] not in self.dirichletNodes:
                            self.dirichletNodes = np.append(self.dirichletNodes, self.subDomains[k][i])
                    elif key[0] == 1:
                        faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
                        self.system[self.subDomains[k][i], self.subDomains[k][:]] += np.matmul(faceNormal[0]*self.fx[k][i,:] + faceNormal[1]*self.fy[k][i,:],self.invPnns[k])
                        if self.subDomains[k][i] not in self.neumannNodes:
                            self.neumannNodes = np.append(self.neumannNodes, self.subDomains[k][i])
                    elif key[0] == 2:
                        faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
                        robinCoefs = robinCoefficients()
                        self.system[self.subDomains[k][i], self.subDomains[k][:]] += np.matmul(robinCoefs[0]*self.f[k][i,:] + robinCoefs[1]*(faceNormal[0]*self.fx[k][i,:] + faceNormal[1]*self.fy[k][i,:]),self.invPnns[k])
                        if self.subDomains[k][i] not in self.robinNodes:
                            self.robinNodes = np.append(self.robinNodes, self.subDomains[k][i])
                    else:
                        self.system[self.subDomains[k][i], self.subDomains[k][:]] += np.matmul(self.f[k][i,:] - dt*materialCoefficient*(self.fxx[k][i,:] + self.fyy[k][i,:]),self.invPnns[k])

        elif timeScheme == 'forwardEuler':
            self.lSys = np.array(np.zeros(self.subDomNo),dtype=object)
            for k in range(self.subDomNo):
                self.lSys[k] = np.zeros((self.subDomains[k].shape[0],self.subDomains[k].shape[0]))
                for i in range(self.subDomains[k].shape[0]):
                    key = onWhichFace(self.mesh.faces, self.boundaries, self.subDomains[k][i])
                    if key[0] == 0:
                        self.lSys[k][i, :] = self.f[k][i,:]
                        if self.subDomains[k][i] not in self.dirichletNodes:
                            self.dirichletNodes = np.append(self.dirichletNodes, self.subDomains[k][i])
                    elif key[0] == 1:
                        faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
                        self.lSys[k][i, :] = faceNormal[0]*self.fx[k][i,:] + faceNormal[1]*self.fy[k][i,:]
                        if self.subDomains[k][i] not in self.neumannNodes:
                            self.neumannNodes = np.append(self.neumannNodes, self.subDomains[k][i])
                    elif key[0] == 2:
                        faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
                        robinCoefs = robinCoefficients()
                        self.lSys[k][i, :] = robinCoefs[0]*self.f[k][i,:] + robinCoefs[1]*(faceNormal[0]*self.fx[k][i,:] + faceNormal[1]*self.fy[k][i,:])
                        if self.subDomains[k][i] not in self.robinNodes:
                            self.robinNodes = np.append(self.robinNodes, self.subDomains[k][i])
                    else:
                        self.lSys[k][i, :] = dt*materialCoefficient*(self.fxx[k][i,:] + self.fyy[k][i,:])


    def unsteadySolve(self, endTime=1):
        invSystem=np.linalg.pinv(self.system)
        timeStep_total = int(endTime / self.dt)
        x = self.mesh.locations[:,0]
        y = self.mesh.locations[:,1]
        self.soln = initialCondition(x,y)

        if self.timeScheme == 'backwardEuler':
            for time in range(timeStep_total):
                self.rhs = np.zeros(self.mesh.nodeNo)
                for k in range(self.subDomNo):
                    for i in range(self.subDomains[k].shape[0]):
                        key = onWhichFace(self.mesh.faces, self.boundaries, self.subDomains[k][i])
                        if key[0] == 0:
                            self.rhs[self.subDomains[k][i]] += unknown(self.mesh.locations[self.subDomains[k][i],0], self.mesh.locations[self.subDomains[k][i],1], (time+1)*self.dt)
                        elif key[0] == 1:
                            self.rhs[self.subDomains[k][i]] += du_dn(self.mesh.locations[self.subDomains[k][i],0], self.mesh.locations[self.subDomains[k][i],1], (time+1)*self.dt)
                        elif key[0] == 2:
                            self.rhs[self.subDomains[k][i]] += robin(self.mesh.locations[self.subDomains[k][i],0], self.mesh.locations[self.subDomains[k][i],1], (time+1)*self.dt)
                        else:
                            self.rhs[self.subDomains[k][i]] += self.dt*source(self.mesh.locations[self.subDomains[k][i], 0], self.mesh.locations[self.subDomains[k][i], 1], (time+1)*self.dt) + self.soln[self.subDomains[k][i]]
                self.soln = np.matmul(invSystem,self.rhs)

class LRBFCM(object):
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
        self.subDomainRadius = self.mesh.spatialSteps[0]
        self.boundaries = boundaries

        boundaryNodeNo = self.mesh.boundaryNodeNo
        subDomNo = self.mesh.nodeNo - boundaryNodeNo
        self.subDomNo = subDomNo
        epsilon=1e-14
        self.subDomains = np.array(np.zeros(subDomNo),dtype=object)
        jj = 0
        for i in self.mesh.interior:
            xi, yi = self.mesh.locations[i,0], self.mesh.locations[i,1]
            influencer = np.array([], dtype=int)
            for j in range(self.mesh.nodeNo+self.mesh.interfaceBNN):
                xj, yj = self.mesh.locations[j,0], self.mesh.locations[j,1]
                distance = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if distance<=self.subDomainRadius+epsilon:
                    influencer = np.append(influencer, j)
            self.subDomains[jj] = influencer
            jj += 1

        dummy = np.array([])
        for i in range(subDomNo):
            dummy = np.concatenate([dummy, self.subDomains[i]])
        holder = getMissing(self.mesh.nodeNo, dummy)
        for k in range(subDomNo):
            for i in range(self.subDomains[k].size):
                checker = self.subDomains[k][i]
                for ii in holder:
                    if checker>ii:
                        self.subDomains[k][i] = self.subDomains[k][i] - 1
        for i in range(self.mesh.interior.size):
            checker = self.mesh.interior[i]
            for ii in holder:
                if checker>ii:
                    self.mesh.interior[i] = self.mesh.interior[i] - 1
        for k in range(self.mesh.faces.size):
            j = 0
            for i in self.mesh.faces[k]:
                for ii in holder:
                    if i == ii:
                        self.mesh.faces[k] = np.delete(self.mesh.faces[k], j, 0)
                j += 1
        for k in range(self.mesh.faces.size):
            j = 0
            for i in self.mesh.faces[k]:
                for ii in holder:
                    if i > ii:
                        self.mesh.faces[k][j] = self.mesh.faces[k][j] - 1
                j += 1
        holder_mod = holder
        for i in range(holder.size):
            holder_mod[i] = holder[i] - i
        for i in holder_mod:
            self.mesh.locations = np.delete(self.mesh.locations, i, 0)
            self.mesh.nodeNo = self.mesh.nodeNo - 1
            self.mesh.boundaryNodeNo = self.mesh.boundaryNodeNo - 1



        self.f = np.array(np.zeros(subDomNo),dtype=object)
        self.fx = np.array(np.zeros(subDomNo),dtype=object)
        self.fxx = np.array(np.zeros(subDomNo),dtype=object)
        self.fy = np.array(np.zeros(subDomNo),dtype=object)
        self.fyy = np.array(np.zeros(subDomNo),dtype=object)
        self.invPnns = np.array(np.zeros(subDomNo), dtype=object)

        if self.rbf == 'MQ':

            for k in range(subDomNo):
                subDomNodeNo = self.subDomains[k].shape[0]
                phiHat = np.zeros((subDomNodeNo,subDomNodeNo))
                phiHat_x = np.zeros((subDomNodeNo,subDomNodeNo))
                phiHat_y = np.zeros((subDomNodeNo,subDomNodeNo))
                phiHat_xx = np.zeros((subDomNodeNo,subDomNodeNo))
                phiHat_yy = np.zeros((subDomNodeNo,subDomNodeNo))
                for i in range(subDomNodeNo):
                    for j in range(subDomNodeNo):
                        r_sq = (self.mesh.locations[self.subDomains[k][i],0] - self.mesh.locations[self.subDomains[k][j],0])**2 + (self.mesh.locations[self.subDomains[k][i],1] - self.mesh.locations[self.subDomains[k][j],1])**2
                        phiHat[i,j] = np.sqrt( r_sq + self.shapeParam**2 )
                        phiHat_x[i, j] = (self.mesh.locations[self.subDomains[k][i],0] - self.mesh.locations[self.subDomains[k][j],0]) / phiHat[i, j]
                        phiHat_y[i, j] = (self.mesh.locations[self.subDomains[k][i],1] - self.mesh.locations[self.subDomains[k][j],1]) / phiHat[i, j]
                        phiHat_xx[i, j] = 1 / phiHat[i, j] - (self.mesh.locations[self.subDomains[k][i],0] - self.mesh.locations[self.subDomains[k][j],0]) ** 2 / phiHat[i, j] ** 3
                        phiHat_yy[i, j] = 1 / phiHat[i, j] - (self.mesh.locations[self.subDomains[k][i],1] - self.mesh.locations[self.subDomains[k][j],1]) ** 2 / phiHat[i, j] ** 3
                self.f[k] = phiHat
                self.fx[k] = phiHat_x
                self.fy[k] = phiHat_y
                self.fxx[k] = phiHat_xx
                self.fyy[k] = phiHat_yy
                self.invPnns[k] = np.linalg.pinv(self.f[k])

        elif self.rbf == 'TPS':
            if self.shapeParam % 2 == 0:
                for k in range(subDomNo):
                    subDomNodeNo = self.subDomains[k].shape[0]
                    phiHat = np.zeros((subDomNodeNo, subDomNodeNo))
                    phiHat_x = np.zeros((subDomNodeNo, subDomNodeNo))
                    phiHat_y = np.zeros((subDomNodeNo, subDomNodeNo))
                    phiHat_xx = np.zeros((subDomNodeNo, subDomNodeNo))
                    phiHat_yy = np.zeros((subDomNodeNo, subDomNodeNo))
                    for i in range(subDomNodeNo):
                        for j in range(subDomNodeNo):
                            if i != j:
                                r = np.sqrt( (self.mesh.locations[self.subDomains[k][i],0] - self.mesh.locations[self.subDomains[k][j],0])**2 + (self.mesh.locations[self.subDomains[k][i],1] - self.mesh.locations[self.subDomains[k][j],1])**2 )
                                phiHat[i,j] = r**self.shapeParam*np.log(r)
                                phiHat_x[i, j] = (self.mesh.locations[self.subDomains[k][i],0]-self.mesh.locations[self.subDomains[k][j],0])*r**(self.shapeParam-2)*(self.shapeParam*np.log(r)+1)
                                phiHat_y[i, j] = (self.mesh.locations[self.subDomains[k][i],1]-self.mesh.locations[self.subDomains[k][j],1])*r**(self.shapeParam-2)*(self.shapeParam*np.log(r)+1)
                                phiHat_xx[i, j] = r**(self.shapeParam-2)*(self.shapeParam*np.log(r)+1)+(self.mesh.locations[self.subDomains[k][i],0]-self.mesh.locations[self.subDomains[k][j],0])**2*r**(self.shapeParam-4)*(2*(self.shapeParam-1)+self.shapeParam*(self.shapeParam-2)*np.log(r))
                                phiHat_yy[i, j] = r**(self.shapeParam-2)*(self.shapeParam*np.log(r)+1)+(self.mesh.locations[self.subDomains[k][i],1]-self.mesh.locations[self.subDomains[k][j],1])**2*r**(self.shapeParam-4)*(2*(self.shapeParam-1)+self.shapeParam*(self.shapeParam-2)*np.log(r))
                    self.f[k] = phiHat
                    self.fx[k] = phiHat_x
                    self.fy[k] = phiHat_y
                    self.fxx[k] = phiHat_xx
                    self.fyy[k] = phiHat_yy
                    self.invPnns[k] = np.linalg.pinv(self.f[k])
            else:
                raise Exception('Enter a positive even number as a shape parameter for TPS.')
        else:
            raise Exception('The radial basis function entered is not defined.')

        self.dirichletNodes = np.array([], dtype=int)
        self.neumannNodes = np.array([], dtype=int)
        self.robinNodes = np.array([], dtype=int)


    def poisson(self, mode, dt=1e-3, endTime=1, tolerance=1e-6, iterLim=3000, materialCoefficient=1):
        self.invBcSys = np.array(np.zeros(self.subDomNo),dtype=object)
        for k in range(self.subDomNo):
            """I have to optimize this part for large number of nodes. All the nodes are considered here. To optimize, mark boundary nodes seperately and consider only boundaries here."""
            bcSys = np.zeros((self.subDomains[k].shape[0],self.subDomains[k].shape[0]))
            for i in range(self.subDomains[k].shape[0]):
                key = onWhichFace(self.mesh.faces, self.boundaries, self.subDomains[k][i])
                if key[0] == 0:
                    bcSys[i, :] = self.f[k][i,:]
                    if self.subDomains[k][i] not in self.dirichletNodes:
                        self.dirichletNodes = np.append(self.dirichletNodes, self.subDomains[k][i])
                elif key[0] == 1:
                    faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
                    bcSys[i, :] = faceNormal[0]*self.fx[k][i,:] + faceNormal[1]*self.fy[k][i,:]
                    if self.subDomains[k][i] not in self.neumannNodes:
                        self.neumannNodes = np.append(self.neumannNodes, self.subDomains[k][i])
                elif key[0] == 2:
                    faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
                    robinCoefs = robinCoefficients()
                    bcSys[i, :] = robinCoefs[0]*self.f[k][i,:] + robinCoefs[1]*(faceNormal[0]*self.fx[k][i,:] + faceNormal[1]*self.fy[k][i,:])
                    if self.subDomains[k][i] not in self.robinNodes:
                        self.robinNodes = np.append(self.robinNodes, self.subDomains[k][i])
                else:
                    bcSys[i, :] = self.f[k][i,:]
            self.invBcSys[k] = np.linalg.pinv(bcSys)

        self.soln = np.zeros(self.mesh.locations.shape[0])
        for i in range(self.mesh.nodeNo):
            if i in self.dirichletNodes:
                self.soln[i] = unknown(self.mesh.locations[i,0], self.mesh.locations[i,1])
            else:
                self.soln[i] = initialCondition(self.mesh.locations[i,0], self.mesh.locations[i,1])

        alpha = np.array(np.zeros(self.subDomNo), dtype=object)
        localSoln = np.array(np.zeros(self.subDomNo), dtype=object)
        if mode == 'transient':
            total_timeStep = int(endTime/dt)
            for time in range(total_timeStep):
                for k in range(self.subDomNo):
                    localSoln[k] = np.zeros(self.subDomains[k].shape[0])
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln[k][i] = self.soln[self.subDomains[k][i]]
                    alpha[k] = np.matmul(self.invPnns[k],localSoln[k])

                for i in range(self.mesh.boundaryNodeNo, self.mesh.nodeNo):
                    ind = np.where(self.subDomains[i-self.mesh.boundaryNodeNo]==i)
                    self.soln[i] += dt*materialCoefficient*np.matmul(self.fxx[i-self.mesh.boundaryNodeNo][ind,:] + self.fyy[i-self.mesh.boundaryNodeNo][ind,:], alpha[i-self.mesh.boundaryNodeNo]) + dt*source(self.mesh.locations[i,0], self.mesh.locations[i,1], time*dt)

                for k in range(self.subDomNo):
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln[k][i] = self.soln[self.subDomains[k][i]]

                for i in np.append(self.neumannNodes,self.robinNodes):
                    for k in range(self.subDomNo):
                        if i in self.subDomains[k]:
                            ind = np.where(self.subDomains[k]==i)
                            break
                    bcRhs = np.zeros(self.subDomains[k].shape[0])
                    for j in range(self.subDomains[k].shape[0]):
                        if self.subDomains[k][j] in self.dirichletNodes:
                            bcRhs[j] = localSoln[k][j]
                        elif self.subDomains[k][j] in self.neumannNodes:
                            bcRhs[j] = du_dn(self.mesh.locations[self.subDomains[k][j],0], self.mesh.locations[self.subDomains[k][j],1], (time+1)*dt)
                        elif self.subDomains[k][j] in self.robinNodes:
                            bcRhs[j] = robin(self.mesh.locations[self.subDomains[k][j],0], self.mesh.locations[self.subDomains[k][j],1], (time+1)*dt)
                        else:
                            bcRhs[j] = localSoln[k][j]
                    alphaBc = np.matmul(self.invBcSys[k],bcRhs)
                    self.soln[i] = np.matmul(self.f[k][ind,:], alphaBc)

                print('Time:    ', (time+1)*dt)

        elif mode == 'iterative':
            iter = 0
            residual = 1
            dummy = np.zeros(self.soln.size)
            while residual>tolerance and iter<iterLim:
                dummy[:] = self.soln[:]
                for k in range(self.subDomNo):
                    localSoln[k] = np.zeros(self.subDomains[k].shape[0])
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln[k][i] = self.soln[self.subDomains[k][i]]
                    alpha[k] = np.matmul(self.invPnns[k],localSoln[k])

                for i in range(self.mesh.boundaryNodeNo, self.mesh.nodeNo):
                    ind = np.where(self.subDomains[i-self.mesh.boundaryNodeNo]==i)
                    self.soln[i] += dt*materialCoefficient*np.matmul(self.fxx[i-self.mesh.boundaryNodeNo][ind,:] + self.fyy[i-self.mesh.boundaryNodeNo][ind,:], alpha[i-self.mesh.boundaryNodeNo]) + dt*source(self.mesh.locations[i,0], self.mesh.locations[i,1])

                for k in range(self.subDomNo):
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln[k][i] = self.soln[self.subDomains[k][i]]

                for i in np.append(self.neumannNodes,self.robinNodes):
                    for k in range(self.subDomNo):
                        if i in self.subDomains[k]:
                            ind = np.where(self.subDomains[k]==i)
                            break
                    bcRhs = np.zeros(self.subDomains[k].shape[0])
                    for j in range(self.subDomains[k].shape[0]):
                        if self.subDomains[k][j] in self.dirichletNodes:
                            bcRhs[j] = localSoln[k][j]
                        elif self.subDomains[k][j] in self.neumannNodes:
                            bcRhs[j] = du_dn(self.mesh.locations[self.subDomains[k][j],0], self.mesh.locations[self.subDomains[k][j],1])
                        elif self.subDomains[k][j] in self.robinNodes:
                            bcRhs[j] = robin(self.mesh.locations[self.subDomains[k][j],0], self.mesh.locations[self.subDomains[k][j],1])
                        else:
                            bcRhs[j] = localSoln[k][j]
                    alphaBc = np.matmul(self.invBcSys[k],bcRhs)
                    self.soln[i] = np.matmul(self.f[k][ind,:], alphaBc)

                residual = np.max(np.abs(dummy-self.soln))
                iter += 1
                print(iter, '|    Residual: ', residual,'\n')

    def interfacePoisson(self, mode, g2, dt=1e-3, endTime=1, tolerance=1e-6, iterLim=3000, beta2=1, beta1=1, k1=0, k2=0):
        epsilon=1e-14
        self.subDomains_ifc = np.array(np.zeros(self.mesh.interfaceInterior),dtype=object)
        jj = 0
        for i in range(self.mesh.interfaceBNN, self.mesh.interfaceBNN+self.mesh.interfaceInterior):
            xi, yi = self.mesh.interfaceLocs[i,0], self.mesh.interfaceLocs[i,1]
            influencer_ifc = np.array([], dtype=int)
            for j in range(self.mesh.interfaceBNN+self.mesh.interfaceInterior):
                xj, yj = self.mesh.interfaceLocs[j,0], self.mesh.interfaceLocs[j,1]
                distance = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if distance<=self.subDomainRadius+epsilon:
                    influencer_ifc = np.append(influencer_ifc, j)
            self.subDomains_ifc[jj] = influencer_ifc
            jj += 1

        self.solnOuter = np.zeros(self.mesh.nodeNo+self.mesh.interfaceBNN)
        self.solnInner = np.zeros(self.mesh.interfaceInterior + self.mesh.interfaceBNN)
        self.solnOuter[self.mesh.boundaryNodeNo:] = interfaceOuterInterior(self.mesh.locations[self.mesh.boundaryNodeNo:,0],self.mesh.locations[self.mesh.boundaryNodeNo:,1])
        self.solnInner[self.mesh.interfaceBNN:] = interfaceInnerInterior(self.mesh.interfaceLocs[self.mesh.interfaceBNN:,0],self.mesh.interfaceLocs[self.mesh.interfaceBNN:,1])

        self.f_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fx_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fxx_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fy_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fyy_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.invPnns_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        for k in range(self.subDomains_ifc.shape[0]):
            subDomNodeNo = self.subDomains_ifc[k].shape[0]
            phiHat = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_x = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_y = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_xx = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_yy = np.zeros((subDomNodeNo,subDomNodeNo))
            for i in range(subDomNodeNo):
                for j in range(subDomNodeNo):
                    r_sq = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0])**2 + (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1])**2
                    phiHat[i,j] = np.sqrt( r_sq + self.shapeParam**2 )
                    phiHat_x[i, j] = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0]) / phiHat[i, j]
                    phiHat_y[i, j] = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1]) / phiHat[i, j]
                    phiHat_xx[i, j] = 1 / phiHat[i, j] - (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0]) ** 2 / phiHat[i, j] ** 3
                    phiHat_yy[i, j] = 1 / phiHat[i, j] - (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1]) ** 2 / phiHat[i, j] ** 3
            self.f_ifc[k] = phiHat
            self.fx_ifc[k] = phiHat_x
            self.fy_ifc[k] = phiHat_y
            self.fxx_ifc[k] = phiHat_xx
            self.fyy_ifc[k] = phiHat_yy
            self.invPnns_ifc[k] = np.linalg.pinv(self.f_ifc[k])

            self.invBcSys = np.array(np.zeros(self.subDomNo),dtype=object)

        n_out = np.zeros((self.mesh.interfaceBNN,2))
        if self.mesh.shape == 'deltoid':
            n_out[:,0] = -3*self.mesh.r*(np.sin(self.mesh.theta[:]))**2*np.cos(self.mesh.theta[:])
            n_out[:,1] = -3*self.mesh.r*(np.cos(self.mesh.theta[:]))**2*np.sin(self.mesh.theta[:])
        elif self.mesh.shape == 'cercle':
            n_out[:,0] = -self.mesh.r*np.sin(self.mesh.theta[:])
            n_out[:,1] = -self.mesh.r*np.cos(self.mesh.theta[:])

            # tt = 2*i*np.pi/self.mesh.interfaceBNN
            # if tt != 0 and tt != np.pi/2 and tt != np.pi and tt != 3/2*np.pi:
            #     n_out[i,0] = -3*self.mesh.r*(np.sin(tt))**2*np.cos(tt)
            #     n_out[i,1] = -3*self.mesh.r*(np.cos(tt))**2*np.sin(tt)
            #     n_mag = np.sqrt(n_out[i,0]**2+n_out[i,1]**2)
            #     n_out[i,0] = n_out[i,0] / n_mag
            #     n_out[i,1] = n_out[i,1] / n_mag
            # elif tt == 0:
            #     n_out[i,0] = -1
            #     n_out[i,1] = 0
            # elif tt == np.pi/2:
            #     n_out[i,0] = 0
            #     n_out[i,1] = -1
            # elif tt == np.pi:
            #     n_out[i,0] = 1
            #     n_out[i,1] = 0
            # elif tt == 3/2*np.pi:
            #     n_out[i,0] = 0
            #     n_out[i,1] = 1

        for k in range(self.subDomNo):
            """I have to optimize this part for large number of nodes. All the nodes are considered here. To optimize, mark boundary nodes seperately and consider only boundaries here."""
            bcSys = np.zeros((self.subDomains[k].shape[0],self.subDomains[k].shape[0]))
            for i in range(self.subDomains[k].shape[0]):
                if self.subDomains[k][i] < self.mesh.boundaryNodeNo:
                    bcSys[i, :] = self.f[k][i,:]
                elif self.subDomains[k][i] >= self.mesh.nodeNo:
                    bcSys[i, :] = beta2*(n_out[self.subDomains[k][i]-self.mesh.nodeNo,0]*self.fx[k][i,:] + n_out[self.subDomains[k][i]-self.mesh.nodeNo,1]*self.fy[k][i,:])
                else:
                    bcSys[i, :] = self.f[k][i,:]
            self.invBcSys[k] = np.linalg.pinv(bcSys)

        alphaOuter = np.array(np.zeros(self.subDomNo), dtype=object)
        alphaInner = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        localSoln_out = np.array(np.zeros(self.subDomNo), dtype=object)
        localSoln_in = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        v1_x = np.zeros(self.mesh.interfaceBNN+self.mesh.interfaceInterior)
        v1_y = np.zeros(self.mesh.interfaceBNN+self.mesh.interfaceInterior)
        if mode == 'transient':
            total_timeStep = int(endTime/dt)

            self.solnOuter[:self.mesh.boundaryNodeNo] = interfaceOuterBC(self.mesh.locations[:self.mesh.boundaryNodeNo,0],self.mesh.locations[:self.mesh.boundaryNodeNo,1])
            self.solnInner[:self.mesh.interfaceBNN] = interfaceInnerBC(self.solnOuter[self.mesh.nodeNo:],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,0],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,1])
            for time in range(total_timeStep):
                for k in range(self.subDomNo):
                    localSoln_out[k] = np.zeros(self.subDomains[k].shape[0])
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln_out[k][i] = self.solnOuter[self.subDomains[k][i]]
                    alphaOuter[k] = np.matmul(self.invPnns[k],localSoln_out[k])

                for i in self.mesh.interior:
                    ind = np.where(self.subDomains[i-self.mesh.boundaryNodeNo]==i)
                    self.solnOuter[i] += dt*beta2*np.matmul(self.fxx[i-self.mesh.boundaryNodeNo][ind,:] + self.fyy[i-self.mesh.boundaryNodeNo][ind,:], alphaOuter[i-self.mesh.boundaryNodeNo]) + dt*source_f2(self.mesh.locations[i,0], self.mesh.locations[i,1],time*dt) + dt*k2*self.solnOuter[i]

                for k in range(self.subDomNo):
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln_out[k][i] = self.solnOuter[self.subDomains[k][i]]

                for k in range(self.subDomains_ifc.shape[0]):
                    localSoln_in[k] = np.zeros(self.subDomains_ifc[k].shape[0])
                    for i in range(self.subDomains_ifc[k].shape[0]):
                        localSoln_in[k][i] = self.solnInner[self.subDomains_ifc[k][i]]
                    alphaInner[k] = np.matmul(self.invPnns_ifc[k],localSoln_in[k])

                for i in range(self.mesh.interfaceBNN):
                    for k in range(self.subDomains_ifc.shape[0]):
                        if i in self.subDomains_ifc[k]:
                            ind = np.where(self.subDomains_ifc[k]==i)
                            break
                    v1_x[i] = np.matmul(self.fx_ifc[k][ind,:],alphaInner[k])
                    v1_y[i] = np.matmul(self.fy_ifc[k][ind,:],alphaInner[k])

                for i in range(self.mesh.nodeNo, self.mesh.nodeNo+self.mesh.interfaceBNN):
                    for k in range(self.subDomNo):
                        if i in self.subDomains[k]:
                            ind = np.where(self.subDomains[k]==i)
                            break
                    bcRhs = np.zeros(self.subDomains[k].shape[0])
                    for j in range(self.subDomains[k].shape[0]):
                        if self.subDomains[k][j] < self.mesh.nodeNo:
                            bcRhs[j] = localSoln_out[k][j]
                        else:
                            bcRhs[j] = g2 + beta1*(-n_out[i-self.mesh.nodeNo,0]*v1_x[i-self.mesh.nodeNo] - n_out[i-self.mesh.nodeNo,1]*v1_y[i-self.mesh.nodeNo])
                    alphaBc = np.matmul(self.invBcSys[k],bcRhs)
                    self.solnOuter[i] = np.matmul(self.f[k][ind,:], alphaBc)

                for i in range(self.mesh.interfaceBNN, self.mesh.interfaceBNN+self.mesh.interfaceInterior):
                    ind = np.where(self.subDomains_ifc[i-self.mesh.interfaceBNN]==i)
                    self.solnInner[i] += dt*beta1*np.matmul(self.fxx_ifc[i-self.mesh.interfaceBNN][ind,:] + self.fyy_ifc[i-self.mesh.interfaceBNN][ind,:], alphaInner[i-self.mesh.interfaceBNN]) + dt*source_f2(self.mesh.interfaceLocs[i,0], self.mesh.interfaceLocs[i,1],time*dt) + dt*k1*self.solnInner[i]

                self.solnOuter[:self.mesh.boundaryNodeNo] = interfaceOuterBC(self.mesh.locations[:self.mesh.boundaryNodeNo,0],self.mesh.locations[:self.mesh.boundaryNodeNo,1], time*dt)
                self.solnInner[:self.mesh.interfaceBNN] = interfaceInnerBC(self.solnOuter[self.mesh.nodeNo:],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,0],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,1],time*dt)

                print('Time:    ', (time+1)*dt)

        elif mode == 'iterative':
            null = 0

    def interfacePoissonDbcTest(self, mode, g2, dt=1e-3, endTime=1, tolerance=1e-6, iterLim=3000, beta2=1, beta1=1, k1=0, k2=0):
        epsilon=1e-14
        self.subDomains_ifc = np.array(np.zeros(self.mesh.interfaceInterior),dtype=object)
        jj = 0
        for i in range(self.mesh.interfaceBNN, self.mesh.interfaceBNN+self.mesh.interfaceInterior):
            xi, yi = self.mesh.interfaceLocs[i,0], self.mesh.interfaceLocs[i,1]
            influencer_ifc = np.array([], dtype=int)
            for j in range(self.mesh.interfaceBNN+self.mesh.interfaceInterior):
                xj, yj = self.mesh.interfaceLocs[j,0], self.mesh.interfaceLocs[j,1]
                distance = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if distance<=self.subDomainRadius+epsilon:
                    influencer_ifc = np.append(influencer_ifc, j)
            self.subDomains_ifc[jj] = influencer_ifc
            jj += 1

        self.solnOuter = np.zeros(self.mesh.nodeNo+self.mesh.interfaceBNN)
        self.solnInner = np.zeros(self.mesh.interfaceInterior + self.mesh.interfaceBNN)
        self.solnOuter[self.mesh.boundaryNodeNo:self.mesh.nodeNo] = interfaceOuterInterior(self.mesh.locations[self.mesh.boundaryNodeNo:,0],self.mesh.locations[self.mesh.boundaryNodeNo:,1])
        self.solnInner[self.mesh.interfaceBNN:] = interfaceInnerInterior(self.mesh.interfaceLocs[self.mesh.interfaceBNN:,0],self.mesh.interfaceLocs[self.mesh.interfaceBNN:,1])

        self.f_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fx_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fxx_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fy_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fyy_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.invPnns_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        for k in range(self.subDomains_ifc.shape[0]):
            subDomNodeNo = self.subDomains_ifc[k].shape[0]
            phiHat = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_x = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_y = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_xx = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_yy = np.zeros((subDomNodeNo,subDomNodeNo))
            for i in range(subDomNodeNo):
                for j in range(subDomNodeNo):
                    r_sq = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0])**2 + (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1])**2
                    phiHat[i,j] = np.sqrt( r_sq + self.shapeParam**2 )
                    phiHat_x[i, j] = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0]) / phiHat[i, j]
                    phiHat_y[i, j] = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1]) / phiHat[i, j]
                    phiHat_xx[i, j] = 1 / phiHat[i, j] - (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0]) ** 2 / phiHat[i, j] ** 3
                    phiHat_yy[i, j] = 1 / phiHat[i, j] - (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1]) ** 2 / phiHat[i, j] ** 3
            self.f_ifc[k] = phiHat
            self.fx_ifc[k] = phiHat_x
            self.fy_ifc[k] = phiHat_y
            self.fxx_ifc[k] = phiHat_xx
            self.fyy_ifc[k] = phiHat_yy
            self.invPnns_ifc[k] = np.linalg.pinv(self.f_ifc[k])

            self.invBcSys = np.array(np.zeros(self.subDomNo),dtype=object)


        alphaOuter = np.array(np.zeros(self.subDomNo), dtype=object)
        alphaInner = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        localSoln_out = np.array(np.zeros(self.subDomNo), dtype=object)
        localSoln_in = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        if mode == 'transient':
            total_timeStep = int(endTime/dt)

            self.solnOuter[:self.mesh.boundaryNodeNo] = interfaceOuterBC(self.mesh.locations[:self.mesh.boundaryNodeNo,0],self.mesh.locations[:self.mesh.boundaryNodeNo,1])
            self.solnOuter[self.mesh.nodeNo:] = interfaceOuterBC(self.mesh.locations[self.mesh.nodeNo:,0],self.mesh.locations[self.mesh.nodeNo:,1])
            self.solnInner[:self.mesh.interfaceBNN] = interfaceInnerBC(self.solnOuter[self.mesh.nodeNo:],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,0],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,1])
            for time in range(total_timeStep):
                for k in range(self.subDomNo):
                    localSoln_out[k] = np.zeros(self.subDomains[k].shape[0])
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln_out[k][i] = self.solnOuter[self.subDomains[k][i]]
                    alphaOuter[k] = np.matmul(self.invPnns[k],localSoln_out[k])

                for i in self.mesh.interior:
                    ind = np.where(self.subDomains[i-self.mesh.boundaryNodeNo]==i)
                    self.solnOuter[i] += dt*beta2*np.matmul(self.fxx[i-self.mesh.boundaryNodeNo][ind,:] + self.fyy[i-self.mesh.boundaryNodeNo][ind,:], alphaOuter[i-self.mesh.boundaryNodeNo]) - dt*source_f2(self.mesh.locations[i,0], self.mesh.locations[i,1],time*dt) + dt*k2*self.solnOuter[i]

                for k in range(self.subDomains_ifc.shape[0]):
                    localSoln_in[k] = np.zeros(self.subDomains_ifc[k].shape[0])
                    for i in range(self.subDomains_ifc[k].shape[0]):
                        localSoln_in[k][i] = self.solnInner[self.subDomains_ifc[k][i]]
                    alphaInner[k] = np.matmul(self.invPnns_ifc[k],localSoln_in[k])


                for i in range(self.mesh.interfaceBNN, self.mesh.interfaceBNN+self.mesh.interfaceInterior):
                    ind = np.where(self.subDomains_ifc[i-self.mesh.interfaceBNN]==i)
                    self.solnInner[i] += dt*beta1*np.matmul(self.fxx_ifc[i-self.mesh.interfaceBNN][ind,:] + self.fyy_ifc[i-self.mesh.interfaceBNN][ind,:], alphaInner[i-self.mesh.interfaceBNN]) - dt*source_f2(self.mesh.interfaceLocs[i,0], self.mesh.interfaceLocs[i,1],time*dt) + dt*k1*self.solnInner[i]

                self.solnOuter[:self.mesh.boundaryNodeNo] = interfaceOuterBC(self.mesh.locations[:self.mesh.boundaryNodeNo,0],self.mesh.locations[:self.mesh.boundaryNodeNo,1], time*dt)
                self.solnInner[:self.mesh.interfaceBNN] = interfaceInnerBC(self.solnOuter[self.mesh.nodeNo:],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,0],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,1],time*dt)
                self.solnOuter[self.mesh.nodeNo:] = interfaceOuterBC(self.mesh.locations[self.mesh.nodeNo:,0],self.mesh.locations[self.mesh.nodeNo:,1], time*dt)

                print('Time:    ', (time+1)*dt)

        elif mode == 'iterative':
            null = 0

    def interfacePoissonNbcTest(self, mode, g2, dt=1e-3, endTime=1, tolerance=1e-6, iterLim=3000, beta2=1, beta1=1, k1=0, k2=0):
        epsilon=1e-14
        self.subDomains_ifc = np.array(np.zeros(self.mesh.interfaceInterior),dtype=object)
        jj = 0
        for i in range(self.mesh.interfaceBNN, self.mesh.interfaceBNN+self.mesh.interfaceInterior):
            xi, yi = self.mesh.interfaceLocs[i,0], self.mesh.interfaceLocs[i,1]
            influencer_ifc = np.array([], dtype=int)
            for j in range(self.mesh.interfaceBNN+self.mesh.interfaceInterior):
                xj, yj = self.mesh.interfaceLocs[j,0], self.mesh.interfaceLocs[j,1]
                distance = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if distance<=self.subDomainRadius+epsilon:
                    influencer_ifc = np.append(influencer_ifc, j)
            self.subDomains_ifc[jj] = influencer_ifc
            jj += 1

        self.solnOuter = np.zeros(self.mesh.nodeNo+self.mesh.interfaceBNN)
        self.solnInner = np.zeros(self.mesh.interfaceInterior + self.mesh.interfaceBNN)
        self.solnOuter[self.mesh.boundaryNodeNo:self.mesh.nodeNo] = interfaceOuterInterior(self.mesh.locations[self.mesh.boundaryNodeNo:,0],self.mesh.locations[self.mesh.boundaryNodeNo:,1])
        self.solnInner[self.mesh.interfaceBNN:] = interfaceInnerInterior(self.mesh.interfaceLocs[self.mesh.interfaceBNN:,0],self.mesh.interfaceLocs[self.mesh.interfaceBNN:,1])

        self.f_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fx_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fxx_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fy_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fyy_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.invPnns_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        for k in range(self.subDomains_ifc.shape[0]):
            subDomNodeNo = self.subDomains_ifc[k].shape[0]
            phiHat = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_x = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_y = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_xx = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_yy = np.zeros((subDomNodeNo,subDomNodeNo))
            for i in range(subDomNodeNo):
                for j in range(subDomNodeNo):
                    r_sq = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0])**2 + (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1])**2
                    phiHat[i,j] = np.sqrt( r_sq + self.shapeParam**2 )
                    phiHat_x[i, j] = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0]) / phiHat[i, j]
                    phiHat_y[i, j] = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1]) / phiHat[i, j]
                    phiHat_xx[i, j] = 1 / phiHat[i, j] - (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0]) ** 2 / phiHat[i, j] ** 3
                    phiHat_yy[i, j] = 1 / phiHat[i, j] - (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1]) ** 2 / phiHat[i, j] ** 3
            self.f_ifc[k] = phiHat
            self.fx_ifc[k] = phiHat_x
            self.fy_ifc[k] = phiHat_y
            self.fxx_ifc[k] = phiHat_xx
            self.fyy_ifc[k] = phiHat_yy
            self.invPnns_ifc[k] = np.linalg.pinv(self.f_ifc[k])

            self.invBcSys = np.array(np.zeros(self.subDomNo),dtype=object)

        n_out = np.zeros((self.mesh.interfaceBNN,2))
        if self.mesh.shape == 'deltoid':
            for i in range(self.mesh.theta.size):
                if self.mesh.theta[i]<np.pi/2 or (self.mesh.theta[i]>=np.pi and self.mesh.theta[i]<3*np.pi/2):
                    geomCorrector = -1
                else:
                    geomCorrector = 1
                n_out[i,0] = geomCorrector*np.sin(self.mesh.theta[i])
                n_out[i,1] = geomCorrector*np.cos(self.mesh.theta[i])
        elif self.mesh.shape == 'cercle':
            n_out[:,0] = -np.cos(self.mesh.theta[:])
            n_out[:,1] = -np.sin(self.mesh.theta[:])

        for k in range(self.subDomNo):
            """I have to optimize this part for large number of nodes. All the nodes are considered here. To optimize, mark boundary nodes seperately and consider only boundaries here."""
            bcSys = np.zeros((self.subDomains[k].shape[0],self.subDomains[k].shape[0]))
            for i in range(self.subDomains[k].shape[0]):
                if self.subDomains[k][i] < self.mesh.boundaryNodeNo:
                    bcSys[i, :] = self.f[k][i,:]
                elif self.subDomains[k][i] >= self.mesh.nodeNo:
                    bcSys[i, :] = beta2*(n_out[self.subDomains[k][i]-self.mesh.nodeNo,0]*self.fx[k][i,:] + n_out[self.subDomains[k][i]-self.mesh.nodeNo,1]*self.fy[k][i,:])
                else:
                    bcSys[i, :] = self.f[k][i,:]
            self.invBcSys[k] = np.linalg.pinv(bcSys)

        alphaOuter = np.array(np.zeros(self.subDomNo), dtype=object)
        alphaInner = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        localSoln_out = np.array(np.zeros(self.subDomNo), dtype=object)
        localSoln_in = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        if mode == 'transient':
            total_timeStep = int(endTime/dt)

            self.solnOuter[:self.mesh.boundaryNodeNo] = interfaceOuterBC(self.mesh.locations[:self.mesh.boundaryNodeNo,0],self.mesh.locations[:self.mesh.boundaryNodeNo,1])
            self.solnInner[:self.mesh.interfaceBNN] = interfaceInnerBC(self.solnOuter[self.mesh.nodeNo:],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,0],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,1])
            for time in range(total_timeStep):
                for k in range(self.subDomNo):
                    localSoln_out[k] = np.zeros(self.subDomains[k].shape[0])
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln_out[k][i] = self.solnOuter[self.subDomains[k][i]]
                    alphaOuter[k] = np.matmul(self.invPnns[k],localSoln_out[k])

                for i in self.mesh.interior:
                    ind = np.where(self.subDomains[i-self.mesh.boundaryNodeNo]==i)
                    self.solnOuter[i] += dt*beta2*np.matmul(self.fxx[i-self.mesh.boundaryNodeNo][ind,:] + self.fyy[i-self.mesh.boundaryNodeNo][ind,:], alphaOuter[i-self.mesh.boundaryNodeNo]) + dt*source_f2(self.mesh.locations[i,0], self.mesh.locations[i,1],time*dt) + dt*k2*self.solnOuter[i]

                for k in range(self.subDomNo):
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln_out[k][i] = self.solnOuter[self.subDomains[k][i]]

                for k in range(self.subDomains_ifc.shape[0]):
                    localSoln_in[k] = np.zeros(self.subDomains_ifc[k].shape[0])
                    for i in range(self.subDomains_ifc[k].shape[0]):
                        localSoln_in[k][i] = self.solnInner[self.subDomains_ifc[k][i]]
                    alphaInner[k] = np.matmul(self.invPnns_ifc[k],localSoln_in[k])

                for i in range(self.mesh.nodeNo, self.mesh.nodeNo+self.mesh.interfaceBNN):
                    for k in range(self.subDomNo):
                        if i in self.subDomains[k]:
                            ind = np.where(self.subDomains[k]==i)
                            break
                    bcRhs = np.zeros(self.subDomains[k].shape[0])
                    for j in range(self.subDomains[k].shape[0]):
                        if self.subDomains[k][j] < self.mesh.nodeNo:
                            bcRhs[j] = localSoln_out[k][j]
                        else:
                            bcRhs[j] = -1
                    alphaBc = np.matmul(self.invBcSys[k],bcRhs)
                    self.solnOuter[i] = np.matmul(self.f[k][ind,:], alphaBc)

                for i in range(self.mesh.interfaceBNN, self.mesh.interfaceBNN+self.mesh.interfaceInterior):
                    ind = np.where(self.subDomains_ifc[i-self.mesh.interfaceBNN]==i)
                    self.solnInner[i] += dt*beta1*np.matmul(self.fxx_ifc[i-self.mesh.interfaceBNN][ind,:] + self.fyy_ifc[i-self.mesh.interfaceBNN][ind,:], alphaInner[i-self.mesh.interfaceBNN]) + dt*source_f2(self.mesh.interfaceLocs[i,0], self.mesh.interfaceLocs[i,1],time*dt) + dt*k1*self.solnInner[i]

                self.solnOuter[:self.mesh.boundaryNodeNo] = interfaceOuterBC(self.mesh.locations[:self.mesh.boundaryNodeNo,0],self.mesh.locations[:self.mesh.boundaryNodeNo,1], time*dt)
                self.solnInner[:self.mesh.interfaceBNN] = interfaceInnerBC(self.solnOuter[self.mesh.nodeNo:],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,0],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,1],time*dt)

                print('Time:    ', (time+1)*dt)

        elif mode == 'iterative':
            null = 0

    def interfacePoissonTest(self, mode, g2, dt=1e-3, endTime=1, tolerance=1e-6, iterLim=3000, beta2=1, beta1=1, k1=0, k2=0):
        epsilon=1e-14
        self.subDomains_ifc = np.array(np.zeros(self.mesh.interfaceInterior),dtype=object)
        jj = 0
        for i in range(self.mesh.interfaceBNN, self.mesh.interfaceBNN+self.mesh.interfaceInterior):
            xi, yi = self.mesh.interfaceLocs[i,0], self.mesh.interfaceLocs[i,1]
            influencer_ifc = np.array([], dtype=int)
            for j in range(self.mesh.interfaceBNN+self.mesh.interfaceInterior):
                xj, yj = self.mesh.interfaceLocs[j,0], self.mesh.interfaceLocs[j,1]
                distance = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if distance<=self.subDomainRadius+epsilon:
                    influencer_ifc = np.append(influencer_ifc, j)
            self.subDomains_ifc[jj] = influencer_ifc
            jj += 1

        self.solnOuter = np.zeros(self.mesh.nodeNo+self.mesh.interfaceBNN)
        self.solnInner = np.zeros(self.mesh.interfaceInterior + self.mesh.interfaceBNN)
        self.solnOuter[self.mesh.boundaryNodeNo:self.mesh.nodeNo] = interfaceOuterInterior(self.mesh.locations[self.mesh.boundaryNodeNo:,0],self.mesh.locations[self.mesh.boundaryNodeNo:,1])
        self.solnInner[self.mesh.interfaceBNN:] = interfaceInnerInterior(self.mesh.interfaceLocs[self.mesh.interfaceBNN:,0],self.mesh.interfaceLocs[self.mesh.interfaceBNN:,1])

        self.f_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fx_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fxx_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fy_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fyy_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.invPnns_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        for k in range(self.subDomains_ifc.shape[0]):
            subDomNodeNo = self.subDomains_ifc[k].shape[0]
            phiHat = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_x = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_y = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_xx = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_yy = np.zeros((subDomNodeNo,subDomNodeNo))
            for i in range(subDomNodeNo):
                for j in range(subDomNodeNo):
                    r_sq = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0])**2 + (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1])**2
                    phiHat[i,j] = np.sqrt( r_sq + self.shapeParam**2 )
                    phiHat_x[i, j] = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0]) / phiHat[i, j]
                    phiHat_y[i, j] = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1]) / phiHat[i, j]
                    phiHat_xx[i, j] = 1 / phiHat[i, j] - (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0]) ** 2 / phiHat[i, j] ** 3
                    phiHat_yy[i, j] = 1 / phiHat[i, j] - (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1]) ** 2 / phiHat[i, j] ** 3
            self.f_ifc[k] = phiHat
            self.fx_ifc[k] = phiHat_x
            self.fy_ifc[k] = phiHat_y
            self.fxx_ifc[k] = phiHat_xx
            self.fyy_ifc[k] = phiHat_yy
            self.invPnns_ifc[k] = np.linalg.pinv(self.f_ifc[k])

            self.invBcSys = np.array(np.zeros(self.subDomNo),dtype=object)

        n_out = np.zeros((self.mesh.interfaceBNN,2))
        if self.mesh.shape == 'deltoid':
            for i in range(self.mesh.theta.size):
                if self.mesh.theta[i]<np.pi/2 or (self.mesh.theta[i]>=np.pi and self.mesh.theta[i]<3*np.pi/2):
                    geomCorrector = -1
                else:
                    geomCorrector = 1
                n_out[i,0] = geomCorrector*np.sin(self.mesh.theta[i])
                n_out[i,1] = geomCorrector*np.cos(self.mesh.theta[i])
        elif self.mesh.shape == 'cercle':
            n_out[:,0] = -np.cos(self.mesh.theta[:])
            n_out[:,1] = -np.sin(self.mesh.theta[:])

        for k in range(self.subDomNo):
            """I have to optimize this part for large number of nodes. All the nodes are considered here. To optimize, mark boundary nodes seperately and consider only boundaries here."""
            bcSys = np.zeros((self.subDomains[k].shape[0],self.subDomains[k].shape[0]))
            for i in range(self.subDomains[k].shape[0]):
                if self.subDomains[k][i] < self.mesh.boundaryNodeNo:
                    bcSys[i, :] = self.f[k][i,:]
                elif self.subDomains[k][i] >= self.mesh.nodeNo:
                    bcSys[i, :] = beta2*(n_out[self.subDomains[k][i]-self.mesh.nodeNo,0]*self.fx[k][i,:] + n_out[self.subDomains[k][i]-self.mesh.nodeNo,1]*self.fy[k][i,:])
                else:
                    bcSys[i, :] = self.f[k][i,:]
            self.invBcSys[k] = np.linalg.pinv(bcSys)

        alphaOuter = np.array(np.zeros(self.subDomNo), dtype=object)
        alphaInner = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        localSoln_out = np.array(np.zeros(self.subDomNo), dtype=object)
        localSoln_in = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        v1_x = np.zeros(self.mesh.interfaceBNN)
        v1_y = np.zeros(self.mesh.interfaceBNN)
        if mode == 'transient':
            total_timeStep = int(endTime/dt)

            self.solnOuter[:self.mesh.boundaryNodeNo] = interfaceOuterBC(self.mesh.locations[:self.mesh.boundaryNodeNo,0],self.mesh.locations[:self.mesh.boundaryNodeNo,1])
            self.solnInner[:self.mesh.interfaceBNN] = interfaceInnerBC(self.solnOuter[self.mesh.nodeNo:],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,0],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,1])
            for time in range(total_timeStep):
                for k in range(self.subDomNo):
                    localSoln_out[k] = np.zeros(self.subDomains[k].shape[0])
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln_out[k][i] = self.solnOuter[self.subDomains[k][i]]
                    alphaOuter[k] = np.matmul(self.invPnns[k],localSoln_out[k])

                for i in self.mesh.interior:
                    ind = np.where(self.subDomains[i-self.mesh.boundaryNodeNo]==i)
                    self.solnOuter[i] += dt*beta2*np.matmul(self.fxx[i-self.mesh.boundaryNodeNo][ind,:] + self.fyy[i-self.mesh.boundaryNodeNo][ind,:], alphaOuter[i-self.mesh.boundaryNodeNo]) + dt*source_f2(self.mesh.locations[i,0], self.mesh.locations[i,1],time*dt) + dt*k2*self.solnOuter[i]

                for k in range(self.subDomNo):
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln_out[k][i] = self.solnOuter[self.subDomains[k][i]]

                for k in range(self.subDomains_ifc.shape[0]):
                    localSoln_in[k] = np.zeros(self.subDomains_ifc[k].shape[0])
                    for i in range(self.subDomains_ifc[k].shape[0]):
                        localSoln_in[k][i] = self.solnInner[self.subDomains_ifc[k][i]]
                    alphaInner[k] = np.matmul(self.invPnns_ifc[k],localSoln_in[k])

                for i in range(self.mesh.interfaceBNN, self.mesh.interfaceBNN+self.mesh.interfaceInterior):
                    ind = np.where(self.subDomains_ifc[i-self.mesh.interfaceBNN]==i)
                    self.solnInner[i] += dt*beta1*np.matmul(self.fxx_ifc[i-self.mesh.interfaceBNN][ind,:] + self.fyy_ifc[i-self.mesh.interfaceBNN][ind,:], alphaInner[i-self.mesh.interfaceBNN]) + dt*source_f2(self.mesh.interfaceLocs[i,0], self.mesh.interfaceLocs[i,1],time*dt) + dt*k1*self.solnInner[i]

                for k in range(self.subDomNo):
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln_out[k][i] = self.solnOuter[self.subDomains[k][i]]
                    alphaOuter[k] = np.matmul(self.invPnns[k],localSoln_out[k])

                self.solnInner[:self.mesh.interfaceBNN] = interfaceInnerBC(self.solnOuter[self.mesh.nodeNo:],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,0],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,1],time*dt)

                for k in range(self.subDomains_ifc.shape[0]):
                    for i in range(self.subDomains_ifc[k].shape[0]):
                        localSoln_in[k][i] = self.solnInner[self.subDomains_ifc[k][i]]
                    alphaInner[k] = np.matmul(self.invPnns_ifc[k],localSoln_in[k])

                for i in range(self.mesh.interfaceBNN):
                    for k in range(self.subDomains_ifc.shape[0]):
                        if i in self.subDomains_ifc[k]:
                            ind = np.where(self.subDomains_ifc[k]==i)
                            break
                    v1_x[i] = np.matmul(self.fx_ifc[k][ind,:],alphaInner[k])
                    v1_y[i] = np.matmul(self.fy_ifc[k][ind,:],alphaInner[k])

                for i in range(self.mesh.nodeNo, self.mesh.nodeNo+self.mesh.interfaceBNN):
                    for k in range(self.subDomNo):
                        if i in self.subDomains[k]:
                            ind = np.where(self.subDomains[k]==i)
                            break
                    bcRhs = np.zeros(self.subDomains[k].shape[0])
                    for j in range(self.subDomains[k].shape[0]):
                        if self.subDomains[k][j] < self.mesh.nodeNo:
                            bcRhs[j] = localSoln_out[k][j]
                        else:
                            bcRhs[j] = g2 + beta1*(-n_out[self.subDomains[k][j]-self.mesh.nodeNo,0]*v1_x[self.subDomains[k][j]-self.mesh.nodeNo] - n_out[self.subDomains[k][j]-self.mesh.nodeNo,1]*v1_y[self.subDomains[k][j]-self.mesh.nodeNo])
                    alphaBc = np.matmul(self.invBcSys[k],bcRhs)
                    self.solnOuter[i] = np.matmul(self.f[k][ind,:], alphaBc)

                self.solnOuter[:self.mesh.boundaryNodeNo] = interfaceOuterBC(self.mesh.locations[:self.mesh.boundaryNodeNo,0],self.mesh.locations[:self.mesh.boundaryNodeNo,1], time*dt)

                print('Time:    ', (time+1)*dt)

        elif mode == 'iterative':
            null = 0

    def interfacePoissonMixTest(self, mode, g2, dt=1e-3, endTime=1, tolerance=1e-6, iterLim=3000, beta2=1, beta1=1, k1=0, k2=0):
        epsilon=1e-14
        self.subDomains_ifc = np.array(np.zeros(self.mesh.interfaceInterior),dtype=object)
        jj = 0
        for i in range(self.mesh.interfaceBNN, self.mesh.interfaceBNN+self.mesh.interfaceInterior):
            xi, yi = self.mesh.interfaceLocs[i,0], self.mesh.interfaceLocs[i,1]
            influencer_ifc = np.array([], dtype=int)
            for j in range(self.mesh.interfaceBNN+self.mesh.interfaceInterior):
                xj, yj = self.mesh.interfaceLocs[j,0], self.mesh.interfaceLocs[j,1]
                distance = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if distance<=self.subDomainRadius+epsilon:
                    influencer_ifc = np.append(influencer_ifc, j)
            self.subDomains_ifc[jj] = influencer_ifc
            jj += 1

        self.solnOuter = np.zeros(self.mesh.nodeNo+self.mesh.interfaceBNN)
        self.solnInner = np.zeros(self.mesh.interfaceInterior + self.mesh.interfaceBNN)
        self.solnOuter[self.mesh.boundaryNodeNo:self.mesh.nodeNo] = interfaceOuterInterior(self.mesh.locations[self.mesh.boundaryNodeNo:,0],self.mesh.locations[self.mesh.boundaryNodeNo:,1])
        self.solnInner[self.mesh.interfaceBNN:] = interfaceInnerInterior(self.mesh.interfaceLocs[self.mesh.interfaceBNN:,0],self.mesh.interfaceLocs[self.mesh.interfaceBNN:,1])

        self.f_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fx_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fxx_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fy_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.fyy_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]),dtype=object)
        self.invPnns_ifc = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        for k in range(self.subDomains_ifc.shape[0]):
            subDomNodeNo = self.subDomains_ifc[k].shape[0]
            phiHat = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_x = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_y = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_xx = np.zeros((subDomNodeNo,subDomNodeNo))
            phiHat_yy = np.zeros((subDomNodeNo,subDomNodeNo))
            for i in range(subDomNodeNo):
                for j in range(subDomNodeNo):
                    r_sq = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0])**2 + (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1])**2
                    phiHat[i,j] = np.sqrt( r_sq + self.shapeParam**2 )
                    phiHat_x[i, j] = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0]) / phiHat[i, j]
                    phiHat_y[i, j] = (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1]) / phiHat[i, j]
                    phiHat_xx[i, j] = 1 / phiHat[i, j] - (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],0] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],0]) ** 2 / phiHat[i, j] ** 3
                    phiHat_yy[i, j] = 1 / phiHat[i, j] - (self.mesh.interfaceLocs[self.subDomains_ifc[k][i],1] - self.mesh.interfaceLocs[self.subDomains_ifc[k][j],1]) ** 2 / phiHat[i, j] ** 3
            self.f_ifc[k] = phiHat
            self.fx_ifc[k] = phiHat_x
            self.fy_ifc[k] = phiHat_y
            self.fxx_ifc[k] = phiHat_xx
            self.fyy_ifc[k] = phiHat_yy
            self.invPnns_ifc[k] = np.linalg.pinv(self.f_ifc[k])

            self.invBcSys = np.array(np.zeros(self.subDomNo),dtype=object)

        n_out = np.zeros((self.mesh.interfaceBNN,2))
        if self.mesh.shape == 'deltoid':
            for i in range(self.mesh.theta.size):
                if self.mesh.theta[i]<np.pi/2 or (self.mesh.theta[i]>=np.pi and self.mesh.theta[i]<3*np.pi/2):
                    geomCorrector = -1
                else:
                    geomCorrector = 1
                n_out[i,0] = geomCorrector*np.sin(self.mesh.theta[i])
                n_out[i,1] = geomCorrector*np.cos(self.mesh.theta[i])
        elif self.mesh.shape == 'cercle':
            n_out[:,0] = -np.cos(self.mesh.theta[:])
            n_out[:,1] = -np.sin(self.mesh.theta[:])

        for k in range(self.subDomNo):
            """I have to optimize this part for large number of nodes. All the nodes are considered here. To optimize, mark boundary nodes seperately and consider only boundaries here."""
            bcSys = np.zeros((self.subDomains[k].shape[0],self.subDomains[k].shape[0]))
            for i in range(self.subDomains[k].shape[0]):
                if self.subDomains[k][i] < self.mesh.boundaryNodeNo:
                    bcSys[i, :] = self.f[k][i,:]
                elif self.subDomains[k][i] >= self.mesh.nodeNo:
                    bcSys[i, :] = (beta1+beta2)*(n_out[self.subDomains[k][i]-self.mesh.nodeNo,0]*self.fx[k][i,:] + n_out[self.subDomains[k][i]-self.mesh.nodeNo,1]*self.fy[k][i,:]) #beta2-->(beta1+beta2) assuming v_1 = v_2
                else:
                    bcSys[i, :] = self.f[k][i,:]
            self.invBcSys[k] = np.linalg.pinv(bcSys)

        alphaOuter = np.array(np.zeros(self.subDomNo), dtype=object)
        alphaInner = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        localSoln_out = np.array(np.zeros(self.subDomNo), dtype=object)
        localSoln_in = np.array(np.zeros(self.subDomains_ifc.shape[0]), dtype=object)
        v1_x = np.zeros(self.mesh.interfaceBNN)
        v1_y = np.zeros(self.mesh.interfaceBNN)
        if mode == 'transient':
            total_timeStep = int(endTime/dt)

            self.solnOuter[:self.mesh.boundaryNodeNo] = interfaceOuterBC(self.mesh.locations[:self.mesh.boundaryNodeNo,0],self.mesh.locations[:self.mesh.boundaryNodeNo,1])
            for time in range(total_timeStep):
                for k in range(self.subDomNo):
                    localSoln_out[k] = np.zeros(self.subDomains[k].shape[0])
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln_out[k][i] = self.solnOuter[self.subDomains[k][i]]
                    alphaOuter[k] = np.matmul(self.invPnns[k],localSoln_out[k])

                for i in self.mesh.interior:
                    ind = np.where(self.subDomains[i-self.mesh.boundaryNodeNo]==i)
                    self.solnOuter[i] += dt*beta2*np.matmul(self.fxx[i-self.mesh.boundaryNodeNo][ind,:] + self.fyy[i-self.mesh.boundaryNodeNo][ind,:], alphaOuter[i-self.mesh.boundaryNodeNo]) + dt*source_f2(self.mesh.locations[i,0], self.mesh.locations[i,1],time*dt) + dt*k2*self.solnOuter[i]

                for k in range(self.subDomNo):
                    for i in range(self.subDomains[k].shape[0]):
                        localSoln_out[k][i] = self.solnOuter[self.subDomains[k][i]]

                for k in range(self.subDomains_ifc.shape[0]):
                    localSoln_in[k] = np.zeros(self.subDomains_ifc[k].shape[0])
                    for i in range(self.subDomains_ifc[k].shape[0]):
                        localSoln_in[k][i] = self.solnInner[self.subDomains_ifc[k][i]]
                    alphaInner[k] = np.matmul(self.invPnns_ifc[k],localSoln_in[k])

                for i in range(self.mesh.interfaceBNN, self.mesh.interfaceBNN+self.mesh.interfaceInterior):
                    ind = np.where(self.subDomains_ifc[i-self.mesh.interfaceBNN]==i)
                    self.solnInner[i] += dt*beta1*np.matmul(self.fxx_ifc[i-self.mesh.interfaceBNN][ind,:] + self.fyy_ifc[i-self.mesh.interfaceBNN][ind,:], alphaInner[i-self.mesh.interfaceBNN]) + dt*source_f2(self.mesh.interfaceLocs[i,0], self.mesh.interfaceLocs[i,1],time*dt) + dt*k1*self.solnInner[i]

                for i in range(self.mesh.nodeNo, self.mesh.nodeNo+self.mesh.interfaceBNN):
                    for k in range(self.subDomNo):
                        if i in self.subDomains[k]:
                            ind = np.where(self.subDomains[k]==i)
                            break
                    bcRhs = np.zeros(self.subDomains[k].shape[0])
                    for j in range(self.subDomains[k].shape[0]):
                        if self.subDomains[k][j] < self.mesh.nodeNo:
                            bcRhs[j] = localSoln_out[k][j]
                        else:
                            bcRhs[j] = g2_f(self.mesh.locations[self.subDomains[k][j],0],self.mesh.locations[self.subDomains[k][j],1])
                    alphaBc = np.matmul(self.invBcSys[k],bcRhs)
                    self.solnOuter[i] = np.matmul(self.f[k][ind,:], alphaBc)

                self.solnInner[:self.mesh.interfaceBNN] = interfaceInnerBC(self.solnOuter[self.mesh.nodeNo:],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,0],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,1])

                # if time == 0:
                #     self.solnOuter[self.mesh.nodeNo:] = 1 #cold starting
                #
                # v1_x_old = 300*np.ones(self.mesh.interfaceBNN)
                # v1_y_old = 300*np.ones(self.mesh.interfaceBNN)
                # v1_old = 300*np.ones(self.mesh.interfaceBNN)
                # while True:
                #     self.solnInner[:self.mesh.interfaceBNN] = interfaceInnerBC(self.solnOuter[self.mesh.nodeNo:],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,0],self.mesh.interfaceLocs[:self.mesh.interfaceBNN,1])
                #
                #     for k in range(self.subDomains_ifc.shape[0]):
                #         for i in range(self.subDomains_ifc[k].shape[0]):
                #             localSoln_in[k][i] = self.solnInner[self.subDomains_ifc[k][i]]
                #         alphaInner[k] = np.matmul(self.invPnns_ifc[k],localSoln_in[k])
                #
                #     for i in range(self.mesh.interfaceBNN):
                #         for k in range(self.subDomains_ifc.shape[0]):
                #             if i in self.subDomains_ifc[k]:
                #                 ind = np.where(self.subDomains_ifc[k]==i)
                #                 break
                #         v1_x[i] = np.matmul(self.fx_ifc[k][ind,:],alphaInner[k])
                #         v1_y[i] = np.matmul(self.fy_ifc[k][ind,:],alphaInner[k])
                #
                #     for i in range(self.mesh.nodeNo, self.mesh.nodeNo+self.mesh.interfaceBNN):
                #         for k in range(self.subDomNo):
                #             if i in self.subDomains[k]:
                #                 ind = np.where(self.subDomains[k]==i)
                #                 break
                #         bcRhs = np.zeros(self.subDomains[k].shape[0])
                #         for j in range(self.subDomains[k].shape[0]):
                #             if self.subDomains[k][j] < self.mesh.nodeNo:
                #                 bcRhs[j] = localSoln_out[k][j]
                #             else:
                #                 bcRhs[j] = g2_f(self.mesh.locations[self.subDomains[k][j],0],self.mesh.locations[self.subDomains[k][j],1]) + beta1*(-n_out[self.subDomains[k][j]-self.mesh.nodeNo,0]*v1_x[self.subDomains[k][j]-self.mesh.nodeNo] - n_out[self.subDomains[k][j]-self.mesh.nodeNo,1]*v1_y[self.subDomains[k][j]-self.mesh.nodeNo])
                #         alphaBc = np.matmul(self.invBcSys[k],bcRhs)
                #         self.solnOuter[i] = np.matmul(self.f[k][ind,:], alphaBc)
                #
                #     res1 = np.max(np.abs(self.solnInner[:self.mesh.interfaceBNN] - v1_old))
                #     res2 = np.max(np.abs(v1_x - v1_x_old))
                #     res3 = np.max(np.abs(v1_y - v1_y_old))
                #     epsy = 1e-5
                #     print(res1,res2,res3)
                #     if res1<epsy and res2<epsy and res3<epsy:
                #         break
                #
                #     v1_old[:] = self.solnInner[:self.mesh.interfaceBNN]
                #     v1_x_old[:] = v1_x[:]
                #     v1_y_old[:] = v1_y[:]

                print('Time:    ', (time+1)*dt)

        elif mode == 'iterative':
            null = 0

    def scalarTransport(self, velocityField, dt=1e-3, endTime=1, thermalConductivity=1):
        self.invBcSys = np.array(np.zeros(self.subDomNo),dtype=object)
        for k in range(self.subDomNo):
            """I have to optimize this part for large number of nodes. All the nodes are considered here. To optimize, mark boundary nodes seperately and consider only boundaries here."""
            bcSys = np.zeros((self.subDomains[k].shape[0],self.subDomains[k].shape[0]))
            for i in range(self.subDomains[k].shape[0]):
                key = onWhichFace(self.mesh.faces, self.boundaries, self.subDomains[k][i])
                if key[0] == 0:
                    bcSys[i, :] = self.f[k][i,:]
                    if self.subDomains[k][i] not in self.dirichletNodes:
                        self.dirichletNodes = np.append(self.dirichletNodes, self.subDomains[k][i])
                elif key[0] == 1:
                    faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
                    bcSys[i, :] = faceNormal[0]*self.fx[k][i,:] + faceNormal[1]*self.fy[k][i,:]
                    if self.subDomains[k][i] not in self.neumannNodes:
                        self.neumannNodes = np.append(self.neumannNodes, self.subDomains[k][i])
                elif key[0] == 2:
                    faceNormal = faceNormalNeumann(self.mesh.vertices, key[1])
                    robinCoefs = robinCoefficients()
                    bcSys[i, :] = robinCoefs[0]*self.f[k][i,:] + robinCoefs[1]*(faceNormal[0]*self.fx[k][i,:] + faceNormal[1]*self.fy[k][i,:])
                    if self.subDomains[k][i] not in self.robinNodes:
                        self.robinNodes = np.append(self.robinNodes, self.subDomains[k][i])
                else:
                    bcSys[i, :] = self.f[k][i,:]
            self.invBcSys[k] = np.linalg.pinv(bcSys)

        self.soln = np.zeros(self.mesh.locations.shape[0])
        for i in range(self.mesh.nodeNo):
            if i in self.dirichletNodes:
                self.soln[i] = unknown(self.mesh.locations[i,0], self.mesh.locations[i,1])
            else:
                self.soln[i] = initialCondition(self.mesh.locations[i,0], self.mesh.locations[i,1])

        alpha = np.array(np.zeros(self.subDomNo), dtype=object)
        localSoln = np.array(np.zeros(self.subDomNo), dtype=object)
        total_timeStep = int(endTime/dt)
        for time in range(total_timeStep):
            for k in range(self.subDomNo):
                localSoln[k] = np.zeros(self.subDomains[k].shape[0])
                for i in range(self.subDomains[k].shape[0]):
                    localSoln[k][i] = self.soln[self.subDomains[k][i]]
                alpha[k] = np.matmul(self.invPnns[k],localSoln[k])

            for i in range(self.mesh.boundaryNodeNo, self.mesh.nodeNo):
                ind = np.where(self.subDomains[i-self.mesh.boundaryNodeNo]==i)
                self.soln[i] += -dt*velocityField[i,0]*np.matmul(self.fx[i-self.mesh.boundaryNodeNo][ind,:], alpha[i-self.mesh.boundaryNodeNo]) - dt*velocityField[i,1]*np.matmul(self.fy[i-self.mesh.boundaryNodeNo][ind,:], alpha[i-self.mesh.boundaryNodeNo]) + dt*thermalConductivity*np.matmul(self.fxx[i-self.mesh.boundaryNodeNo][ind,:] + self.fyy[i-self.mesh.boundaryNodeNo][ind,:], alpha[i-self.mesh.boundaryNodeNo]) + dt*source(self.mesh.locations[i,0], self.mesh.locations[i,1], time*dt)
            for k in range(self.subDomNo):
                for i in range(self.subDomains[k].shape[0]):
                    localSoln[k][i] = self.soln[self.subDomains[k][i]]

            for i in np.append(self.neumannNodes,self.robinNodes):
                for k in range(self.subDomNo):
                    if i in self.subDomains[k]:
                        ind = np.where(self.subDomains[k]==i)
                        break
                bcRhs = np.zeros(self.subDomains[k].shape[0])
                for j in range(self.subDomains[k].shape[0]):
                    if self.subDomains[k][j] in self.dirichletNodes:
                        bcRhs[j] = localSoln[k][j]
                    elif self.subDomains[k][j] in self.neumannNodes:
                        bcRhs[j] = du_dn(self.mesh.locations[self.subDomains[k][j],0], self.mesh.locations[self.subDomains[k][j],1], (time+1)*dt)
                    elif self.subDomains[k][j] in self.robinNodes:
                        bcRhs[j] = robin(self.mesh.locations[self.subDomains[k][j],0], self.mesh.locations[self.subDomains[k][j],1], (time+1)*dt)
                    else:
                        bcRhs[j] = localSoln[k][j]
                alphaBc = np.matmul(self.invBcSys[k],bcRhs)
                self.soln[i] = np.matmul(self.f[k][ind,:], alphaBc)

            print('Time:    ', (time+1)*dt)

"""Problem definition"""
def source(x,y,t=0):
    # return 5/4 * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y)
    # return -6*x+2
    return 0  # Sarler test 2
    # a = (-29.16 * np.cos(2.7 + 5.4 * y))/(6 + 6*(0.5 + 3*x)**2) + ((2592*(0.5 + 3*x)**2)/(6 + 6*(0.5 + 3*x)**2)**3 - 108/(6 + 6*(0.5 + 3*x)**2)**2)*(1.25 + np.cos(2.7 + 5.4*y))
    # return -a
    # return -2
    # return -np.sin(np.pi*x) * np.sin(np.pi*y)
    # return (x**2-2)*np.exp(t)
    # return -2*y
def unknown(x,y,t=0):
    # return np.sin(np.pi * x) * np.cos(np.pi * y)
    # return x**2*y
    # return x
    # return y+1
    # return 100
    # return x**2
    # return -(np.sin(np.pi*x) * np.sin(np.pi*y)) / (2*np.pi**2)
    # return x**2*np.exp(t)
    return 0 # Sarler test 2
def du_dn(x,y,t=0):
    return 0  # Sarler test 2
    # return 2
    # return 2*y
def robin(x,y,t=0):
    return -750/52
    # return 2*x**2
def robinCoefficients():
    return np.array([1, 1])
def initialCondition(x,y):
    # return 100*np.ones(x.size)
    # return x**2
    # data = pd.read_csv('records/sarlerInitializer.csv')
    # return data['Approx. soln.'].to_numpy
    # return 1  # Sarler test 2
    eps = 1e-8
    if x<.55+eps and x>.45-eps and y<.55+eps and y>.45-eps:
        return 1
    else:
        return 0

def interfaceOuterBC(x,y,t=0):
    return 10*x
def interfaceOuterInterior(x,y,t=0):
    return 0
def interfaceInnerInterior(x,y,t=0):
    return 0
def interfaceInnerBC(solnOuter_cut,x,y,t=0):
    g1 = 0
    return solnOuter_cut - g1
    # return 0
def source_f1(x,y,t=0):
    return 0
def source_f2(x,y,t=0):
    return 0
def g2_f(x,y):
    return -30*np.cos(x)

"""------------------"""

"""Seperate functions"""
def onWhichFace(faces,boundaries,pointIndex):
    n = int(len(faces))
    for i in range(n):
        if pointIndex in faces[i]:
            return np.array([boundaries[i],i])
    return np.array([3,-1])

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
    epsilon = 1e-14
    dummyPoint = point + np.array([0,0])
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
    for i in range(vertices.shape[0]):
        if np.abs(dummyPoint[0]-vertices[i,0])<= epsilon or np.abs(dummyPoint[1]-vertices[i,1])<= epsilon:
            inOrNot = 0
    return inOrNot

def getMissing(upLimit, array):
    holder = np.array([],dtype=int)
    for i in range(upLimit):
        if i not in array:
            holder = np.append(holder, i)
    return holder

def rootMeanSquare(approx,exact):
    return np.sqrt(np.sum((approx-exact)**2)/approx.size)

def insideDeltoid(xp,yp,r):
    """Deltoid is at the center, thus p:(0,0) is already inside the deltoid."""
    if xp==0 and yp==0:
        return True
    elif xp==0 and yp!=0:
        r_sq = yp**2
        distDeltoid_sq = r**2
    elif xp!=0 and yp==0:
        r_sq = xp**2
        distDeltoid_sq = r**2
    else:
        tp = np.arctan(np.abs((yp/xp))**(1/3))
        r_sq = xp**2+yp**2
        xDeltoid = r*(np.cos(tp))**3
        yDeltoid = r*(np.sin(tp))**3
        distDeltoid_sq = xDeltoid**2+yDeltoid**2
    if distDeltoid_sq>=r_sq:
        return True
    else:
        return False

def insideCercle(xp,yp,r):
    """Circle is at the center, thus p:(0,0) is already inside the deltoid."""
    if xp==0 and yp==0:
        return True
    elif xp==0 and yp!=0:
        r_sq = yp**2
        distDeltoid_sq = r**2
    elif xp!=0 and yp==0:
        r_sq = xp**2
        distDeltoid_sq = r**2
    else:
        tp = np.arctan(np.abs((yp/xp)))
        r_sq = xp**2+yp**2
        xDeltoid = r*(np.cos(tp))
        yDeltoid = r*(np.sin(tp))
        distDeltoid_sq = xDeltoid**2+yDeltoid**2
    if distDeltoid_sq>=r_sq:
        return True
    else:
        return False

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