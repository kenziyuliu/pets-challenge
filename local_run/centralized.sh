#!/bin/bash

# Remove any existing `src/` directory
echo '(intial) removing src/ ...'
rm -r src

# Copy the submission src to `src/`
echo 'copying submission_src/pandemic to src ...'
cp -r  submission_src/pandemic src

# Run the file
python3 local_run/centralized.py

echo '(final) removing src/ ...'
rm -r src