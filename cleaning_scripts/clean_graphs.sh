#!/bin/bash

CURRENT_DIRECTORY=$(basename "$PWD")

if [ "${CURRENT_DIRECTORY}" == "cleaning_scripts" ]; then
    cd ../graphs
    echo "Cleaning graphs..."
    rm -rf *
    echo "Done."
else
    echo "Please execute script in cleaning_scripts directory."
    exit 1
fi
