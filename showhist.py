import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


with open('build/particle_histories.out', 'r') as f:
    lines = f.readlines()
clouds = []

for line in lines:
    descriptions = [p.replace(')', '').replace('(', '').strip() for p in line.split(')(')]
    particles = [
        tuple([float(x) for x in desc.split(',')])
        for desc in descriptions
    ]
    clouds.append(particles)

    x = [p[0] for p in particles]
    y = [p[1] for p in particles]
    t = [p[2] for p in particles]

centers = []
for cloud in clouds:
    x = [p[0] for p in cloud]
    y = [p[1] for p in cloud]
    t = [p[2] for p in cloud]
    centers.append((
        np.mean(x), np.mean(y), np.mean(t)
    ))

fig, ax = plt.subplots()
ax.scatter(
    # range(len(centers)),
    [x for (x,y,t) in centers],
#     # c=[t for (x,y,t) in centers],
#     label='x',
#     color='red',
# )
# ax.scatter(
#     range(len(centers)),
    [y for (x,y,t) in centers],    
    c=[t for (x,y,t) in centers],
    # label='y',
    # color='blue',
)
xl = ax.get_xlim()
yl = ax.get_ylim()
# ax.legend()
# plt.show()



class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, clouds, maxindex=None):
        self.clouds = clouds

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(
            self.fig, self.update, interval=64, 
            init_func=self.setup_plot, blit=True
        )

        self.ax.plot(
            [x for (x,y,t) in centers],
            [y for (x,y,t) in centers],
        )

        if maxindex is not None:
            self.maxindex = maxindex
        else:
            self.maxindex = len(self.clouds)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        self.ax.set_xlim(xl)
        self.ax.set_ylim(yl)

        x, y, t = self.gen(0)

        u = np.cos(t)
        v = np.sin(t)

        self.quiv = self.ax.quiver(x, y, u, v, headwidth=0, width=.0001, alpha=.2)
        self.scat = self.ax.scatter(x, y, animated=True, s=2, color='red', alpha=.5)

        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.quiv, self.scat,

    def gen(self, i):
        cloud = self.clouds[i % self.maxindex]
        x = [p[0] for p in cloud]
        y = [p[1] for p in cloud]
        t = [p[2] for p in cloud]
        return x, y, t

    def update(self, i):
        """Update the scatter plot."""
        print 'Updating to frame %d of %d' % (i, len(self.clouds))
        x, y, t = self.gen(i)
        data = np.vstack([x, y, t])
        u = np.cos(t)
        v = np.sin(t)

        verts = data[:2, :].T
        print verts.shape
        self.quiv.set_offsets(verts)
        print u.shape, v.shape
        self.quiv.set_UVC(u, v)
        
        # Set x and y data...
        self.scat.set_offsets(data[:2, :].T)
        # # Set colors..
        # self.scat.set_array(data[2])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.quiv, self.scat,

    def show(self):
        plt.show()

ani = AnimatedScatter(clouds)
ani.show()

