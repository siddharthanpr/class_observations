import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

class POMDP:

    def __init__(self, n_states, n_actions, n_classes):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_classes = n_classes
        self.P = self.windy_world()



    def windy_world(self):
        allP = []
        for c in xrange(self.n_classes):
            P = np.zeros((self.n_states, self.n_states, self.n_actions))
            tol = 1.0e-10
            dim = np.shape(P)
            for k in xrange(dim[2]):
                p = np.zeros((dim[0], dim[1]))
                for i in xrange(dim[0]):
                    for j in xrange(dim[1]):
                        if k == 0:
                            p[i][j] = 1
                        else:
                            p[i][j] = np.exp(-((i-j)**2)/.5)

                for i in xrange(dim[0]):
                    p[i, :] /= p[i,:].sum()

                for i in xrange(dim[0]):
                    for j in xrange(dim[1]):

                        assert abs(p[i,:].sum() - 1) < tol

                if c != 0: k = self.n_actions-1-k
                P[:,:,k] = p
            allP.append(P)

        return allP

    def sample_trajectory(self, start_state, l):
        current_state = start_state
        c_t = []
        traj = []
        start_class_prob = 0.05
        for i in xrange(l):
            cp = np.array([start_class_prob + (1-2*start_class_prob)*float(i)/l, 1-(start_class_prob + (1-2*start_class_prob)*float(i)/l)])
            # cp = np.random.random(self.n_classes)
            # cp /= cp.sum()
            a = np.random.randint(0, self.n_actions)
            c_t.append(cp)
            traj.append((current_state,a),)
            p = np.zeros(self.n_states)
            for cj in xrange(self.n_classes):
                p += cp[cj] * self.P[cj][current_state,:,a]
            current_state = np.random.choice(self.n_states,1,p = p)[0]
        return traj, c_t

    def learn_model(self, traj, c_t):
        allP1 = []
        allP2 = []

        n_piece = len(traj)/self.n_classes

        sas_counts = defaultdict(lambda: 0, {})
        sas_counts_piece = defaultdict(lambda:defaultdict(lambda:0, {}), {})
        sa_counts_piece = defaultdict(lambda:defaultdict(lambda:0, {}), {})
        beta = defaultdict(lambda: 0, {})
        alpha = defaultdict(lambda:defaultdict(lambda:0, {}), {})
        for t in xrange(len(traj)-1):
            sas_counts[(traj[t][0], traj[t][1], traj[t+1][0])] += 1
            sas_counts_piece[t/n_piece][(traj[t][0], traj[t][1], traj[t+1][0])] += 1
            sa_counts_piece[t/n_piece][traj[t]] += 1
            for j in xrange(self.n_classes):
                # if (traj[t][0], traj[t][1], traj[t+1][0], j) == (0,0,0,0) or (traj[t][0], traj[t][1], traj[t+1][0], j) == (0,0,1,1):
                #     print (traj[t][0], traj[t][1], traj[t+1][0], j), c_t[t][j]
                beta[(traj[t][0], traj[t][1], traj[t+1][0], j)] += c_t[t][j]
                alpha[t/n_piece][traj[t][0],traj[t][1],j] += c_t[t][j]

        keys = beta.keys()
        keys.sort()
        for k in keys:
            print k, beta[k]

        # for sasj in beta:
        #     beta[sasj] /= sas_counts[sasj[:-1]]

        for p in alpha:
            for saj in alpha[p]:
                alpha[p][saj] /= sa_counts_piece[p][saj[:-1]]

        A = defaultdict(lambda:np.zeros((self.n_classes,self.n_classes)), {})
        P_bar = defaultdict(lambda:np.zeros(self.n_classes), {})
        for cj in xrange(self.n_classes):

            P1 = np.zeros((self.n_states, self.n_states, self.n_actions))
            P2 = np.zeros((self.n_states, self.n_states, self.n_actions))
            dim = np.shape(P1)
            for k in xrange(dim[2]):
                for i in xrange(dim[0]):
                    for p in alpha:
                        A[(i,k)][p][cj] = alpha[p][(i,k,cj)]

                    for j in xrange(dim[1]):
                        if (i,j,cj) == (0,0,1):
                            print 'here', beta[(i, k, j, cj)]
                        P1[i,j,k] = beta[(i, k, j, cj)]
                        for p in sa_counts_piece:
                            P_bar[(i, k, j)][p] = sas_counts_piece[p][(i,k,j)]/ float(sa_counts_piece[p][(i,k)])
                    P1[i, :, k] /= P1[i, :, k].sum()


            allP1.append(P1)
            allP2.append(P2)

        for k in xrange(dim[2]):
            for i in xrange(dim[0]):
                A_inv = np.linalg.inv(A[(i,k)])
                for j in xrange(dim[1]):
                    psas = A_inv.dot(P_bar[(i, k, j)])
                    for cj in xrange(len(psas)):

                        allP2[cj][i, j, k] = psas[cj]
        return allP1, allP2

p = POMDP(4,2,2)
d = []
it = []

for i in xrange(100):
    print 'iter ', i
    t,c = p.sample_trajectory(0,int(100000/100.0*(i+1)))

    cl = 1
    a = 0
    print 'sampled'
    p1,p2 = p.learn_model(t,c)

    print 'ground truth'
    print p.P[cl][:,:,a]
    # print 'em'
    # print p1[cl][:,:,a]
    # print 'ema2'
    # print p1[cl][:,:,int(not a)]
    # print 'Importance'
    # print p2[cl][:,:,a]

    print '||truth - importance||'
    print np.linalg.norm(p2[cl][:,:,a] - p.P[cl][:,:,a])
    it.append(i*1000)
    d.append(np.linalg.norm(p2[cl][:,:,a] - p.P[cl][:,:,a]))

plt.figure(1)
plt.plot(it,d)

plt.xlabel("Length of trajectory")
plt.ylabel("Frobenius norm of transition matrix error")
plt.show()
# print '||truth - em||'
# print np.linalg.norm(p1[cl][:,:,a] - p.P[cl][:,:,a])

# p.P = p1
#
#
# cl = 0
# a = 1
# t,c = p.sample_trajectory(0,100000)
#
# cl = 0
# a = 1
# print 'sampled'
# p1,p2 = p.learn_model(t,c)
#
# print 'ground truth'
# print p.P[cl][:,:,a]
# print 'em'
# print p1[cl][:,:,a]
# print 'ema2'
# print p1[cl][:,:,int(not a)]
# print 'Importance'
# print p2[cl][:,:,a]
# print '||truth - importance||'
# print np.linalg.norm(p2[cl][:,:,a] - p.P[cl][:,:,a])
# print '||truth - em||'
# print np.linalg.norm(p1[cl][:,:,a] - p.P[cl][:,:,a])
#
# # print '||ema1 - ema2||'
# # print np.linalg.norm(p1[cl][:,:,a] - p1[cl][:,:,int(not a)])
# #
# # print '||trutha1 - ema2||'
# # print np.linalg.norm(p1[cl][:,:,a] - p.P[cl][:,:,int(not a)])
# #
# # print 'done'
