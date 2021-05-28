Module of workers
==================

Each worker module can be run as a standalone program.
run

$ pfbworkers --help

for a list of available workers.

Documentation for each worker is listed under

$ pfbworkers workername --help

All workers can be run on a distributed scheduler chunked over row and
imaging band (the current limitation being that all images fit onto a single
worker).
