# find python site directory (or create it if it doesn't exist)
SITEDIR=$(python -m site --user-site)
mkdir -p "$SITEDIR"

# specify the path containing the project
MYPATH=$"$HOME/Documents/academia/thesis/modelling-microplastics-transport"

# create new .pth file in site directory
echo "$MYPATH" > "$SITEDIR/microplastics_transport.pth"
