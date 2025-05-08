# Evolution Simulator


This was the result of trying to create some form of evolution simulator in one day. To start it you just run sim.py
This simulation is pretty much failed, as cells don't seem to evolve too much. Theres a lot of reasons that probably contribute to this
You can define your grid size at the bottom of the file, and depending on your screen you may have to change the 1000 // GRID_SIZE to be something around your vertical screen resolution. This will probably run anywhere from moderately to extremely slow depending on your computer

Some controls for the simulation 

Space | Advance one cycle
P | Start automatically cycling 
Y | Toggle the FPS limit while autocycling 
J | Print some stats about the sim (pretty sure some of them are bugged)
T and G | Double and Half the amount of plants that grow
R and F | Increase and Decrease the amount of global radiation (a global mutation factor offset)
E and D | Increase and Decrease the global energy (the amount of energy cells get passively)
Q and A | Increase and Decrease the global decisiveness


A quick and incomplete overview of how everything works 


## The Grid

The grid is pretty simple, pretty much just a list of cell objects. Starting off its a random distribution of live cells and empty cells. 

## Live Cells

Live cells, or just cells in the code, are whats actually evolving. Each one has a brain which is a pretty big mess of a pytorch neural network. The cells brain takes in multiple factors:

1. The color and energy of the 8 cells around it 
2. The color and energy of the 5 cells in its line of sight
3. Its current energy 
4. Three "memory" values 

It then outputs 9 values

1. Four values for each direction the cell can move in
2. Whether it wants to reproduce or not
3. Where it wants to look
4. Its three memory values

There is also some code from attempting to trade the model with reinforcement learning but that was very hastily added and I don't think it actually does much

The cell decides where to go with some logic that uses its decisiveness attribute create a probability distribution over its options, kinda weird. The movement is also slightly bugged as the network tends to have very extreme values. This may be way cells are always bias to moving upwards but idk

When a cell reproduces it makes slight modifications to its weights and other attributes depending on its mutation factor


## Plants 

Plants spawn randomly in available squares until a set amount is reached, they don't do anything and exist to provide easy energy


## The cycle

Each cycle every cell gets to take its actions. All actions take energy to do. If two cells try to occupy the same space, a battle will begin and whichever cell has a higher energy will win, getting half the losing cells energy. Plants will always lose battles


That's all. Sometime I'll try this again