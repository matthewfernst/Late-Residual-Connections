#!/bin/bash

CURRENT_DIRECTORY=$(basename $PWD)

if [ "${CURRENT_DIRECTORY}" == "CleaningScripts" ]; then
    cd ../Graphs
    echo "Cleaning graphs..."
    rm -rf *
    echo "Done."
else
    echo "Please execute script in CleaningScripts directory."
    exit 1
fi
