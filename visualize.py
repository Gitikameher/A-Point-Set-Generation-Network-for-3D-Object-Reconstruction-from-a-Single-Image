import numpy as np
import matplotlib.pyplot as plt



class Visualize:
    def __init__(self, pc_list):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        assert type(pc_list) == list
        assert len(pc_list) > 0
        
        #(assert type(pc_list[i]) == np.ndarray) for i in range(len(pc_list))
        #(assert pc_list[i].shape[1] == 3) for i in range(len(pc_list))
        
        self.pc_list = pc_list
    
    
    def ShowRandom(self):
        '''
        Plots 6 random images from list of point clouds
        
        '''
        #assert type(n) == int
        #assert n > 0
        n = 6
        assert n < len(self.pc_list)
        
        
        
#         idx = np.random.choice(len(self.pc_list), n, replace = False)
        idx = np.arange(6)
        fig = plt.figure(figsize = (10,15))
        for i, idx in enumerate(idx):
            ax = fig.add_subplot(2,3,i+1, projection = '3d')
            ax.scatter(self.pc_list[i][:,0],self.pc_list[i][:,1],self.pc_list[i][:,2], c = 'r')
#             print("Shape is\n")
#             print(self.pc_list[i].shape)
        plt.show()

