# pinn2022

Code for paper published in Communications Biology.



## Inputs

The `inputs` folder contains input files for the first case demonstrated in the paper. 

- `volume.dat` is the volume mesh for the finite element model
- `surface.dat` is the surface mesh for the finite element model where it interacts with the flow
- `s_v_mapping.dat` is the nodal index mapping between surface and volume meshes.
- `eigen.dat` contains the eigenfrequencies and eigenvectors of the finite element model
- `measurements.dat` contains the 2D profile data that is used to calculate shape loss.

## Usage

The code is developed based on PyTorch and is supposed to run on GPUs. To run the included case, simply

```bash
python main.py
```

To run a custom case with the code, prepare the dataset in the same format and change the input parameters in `main.py` accordingly.



