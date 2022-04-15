#!/bin/bash

CURRENT_DIRECTORY=$(basename $PWD)

if [ "${CURRENT_DIRECTORY}" == "CleaningScripts" ]; then
    cd ../Code/Results
    echo "Cleaning results..."
    rm -rf *
    echo "Done."
else
    echo "Please execute script in CleaningScripts directory."
fi