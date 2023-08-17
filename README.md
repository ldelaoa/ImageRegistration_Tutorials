# ImageRegistration_Tutorials
Using SimpleITK - Rigid and Affine, not Deformable

# Image Rigid Registration Notebook 
For 2D images

Download dataset from: https://github.com/ldelaoa/ImageRegistration_Tutorials

More info of the code used here on: https://simpleitk.readthedocs.io/en/master/registrationOverview.html
----------------

Registration can face multiple alignment issues. 
This notebook and dataset displays 8 of them assumming Square 1 is our fixed image:.  
Square 1 - Fixed Image.  
Square 2 - Translation 2D.  
Square 3 - Isometric Scaling.  
Square 4 - Rotation.  
Square 5 - Translation 1D.  
Square 6 - Translation 1D + Rotation.  
Square 7 - Translation 2D + Rotation.  
Square 8 - Isometric Scaling + Rotation.  
Square 9 - Isometric Scaling + Translation 2D.  


Possible Transforms for 2D are:  
    -- Euler2DTransform.  
    -- Similarity2DTransform.  
    -- ScaleTransform.  
    -- TranslationTransform.  

Possible Transforms for 3D are:  
    -- ScaleTransform.  
    -- TranslationTransform.  
    -- Euler3DTransform.  
    -- Similarity3DTransform.  
    -- VersorRigid3DTransform.  
    -- ScaleVersor3DTransform.  
    -- TranslationTransform.  

Other configurable Defaults:  
offset="Diff" or "Fix" or "Mov". To calculate the offset at TranslationTransforms, "Diff" is difference between centroids of moving and target. "Fixed" is centroid of Fixed. "Mov" is centroid of Moving.  

iterations_spec=300  . Maximum number of iteration for all Optimizers.  
lr=1  Learning Rate for all Optimizers.  
minStep=.00001  , Minimum Step for Optimizer: Regular Step Gradient Descent.  
gradientT=1e-7  , GradientMagnitudeTolerance for Optimizer :  Regular Step Gradient Descent.  
convWinSize=10  , Convergence Minimum Value for Optimizer: Gradient Descent.  
convMinVal=1e-7 , Convergence Window Size for Optimizer: : Gradient Descent.  
