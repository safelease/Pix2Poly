#!/bin/bash

mkdir -p ../data
cd ../data

# Download train split
wget -r --no-parent https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/
wget -r --no-parent https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/

# Download valid split
wget -r --no-parent https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/
wget -r --no-parent https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/

# Download test split
wget -r --no-parent https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/
wget -r --no-parent https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map/

mv www.cs.toronto.edu/\~vmnih/data/mass_roads .
rm -r www.cs.toronto.edu
