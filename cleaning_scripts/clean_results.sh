#!/bin/bash

CURRENT_DIRECTORY=$(basename $PWD)

if [ "${CURRENT_DIRECTORY}" == "cleaning_scripts" ]; then
    cd ../code/results
    echo "Cleaning results..."
    rm -rf *
    echo "Done."
else
    echo "Please execute script in cleaning_scripts directory."
fi