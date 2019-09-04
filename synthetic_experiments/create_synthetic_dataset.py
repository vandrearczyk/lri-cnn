import sys
sys.path.insert(0, '../')
from utils import *
import numpy as np
from sklearn import preprocessing

def create_synthetic_dataset(n_samples=500,cube_size=32,template_size=7,density_min=.1,density_max=.5,proportions=[0.3,0.7]):
    '''
    Creates a basic 3D texture synthetic dataset.
    Returns the volumes X and labels y.
    n_samples: number of samples per class (default 500)
    cube_size: size of the cubes, i.e. training andtest volumes (default 32)
    template_size: size of the templates rotated and pasted in the volumes (default 7)
    density_min: minimum density of patterns (default 0.1)
    density_max: maximum density of patterns (default 0.5)
    proportions: proportion of template 1 for the two classes (the proportion of template 2 is 1-p) (default= [0.3,0.7])
    '''
    np.random.seed(seed=0)
    # number of classes (only designed for 2 classes here)
    n_class=2
    # Rotation range 
    rot = 360
    range_rot = [0,rot]
    # Generate empty templates
    template = np.zeros((2,template_size,template_size,template_size))
    # Fill the templates
    # For now a simple line for t1
    template[0,int(template_size/2)-1:int(template_size/2)+1,int(template_size/2)-1:int(template_size/2)+1,:] = 1
    # And a cross for t2
    template[1,int(template_size/2)-1:int(template_size/2)+1,int(template_size/2)-1:int(template_size/2)+1,int(template_size/4):int(3*template_size/4)+1] = 1
    template[1,int(template_size/2)-1:int(template_size/2)+1,int(template_size/4):int(3*template_size/4),int(template_size/2)-1:int(template_size/2)+1] = 1
    # Initialize dataset lists
    X = []
    y = []

    for c in range(n_class):
        for s in range(n_samples):
            # Generate an empty 64x64x64 cube
            cube = np.zeros((cube_size,cube_size,cube_size))
            # Generate random density
            density = np.random.uniform(density_min, density_max)
            # Number of patterns in volume based on the density
            n_templates = int((cube_size**3)/(template_size**3)*density)
            # Crop size after rotation:
            crop_size = int(template_size*np.sqrt(3))
            # place the rotated patterns in the cube
            for t in range(n_templates):
                # random position
                position = np.array([np.random.choice(cube_size),np.random.choice(cube_size),np.random.choice(cube_size)])
                # is it template 1 or 2:
                template_type = np.random.choice(2, p=[proportions[c],1-proportions[c]])
                # Rotate the template 1 or 2
                random_angles = [np.random.uniform(range_rot[0], range_rot[1]) for i in range(3)]
                rotated_template = apply_affine_transform_fixed(template[template_type],random_angles)
                # copy the rotated template in the cube
                cube = copy_template(cube,rotated_template,position)
            X.append(cube)
            y.append(c)
    X = np.expand_dims(np.asarray(X),axis=-1)
    y = np.asarray(y)
    le = preprocessing.LabelEncoder()  
    le.fit(np.unique(y))
    y = le.transform(y) 
    return X,y