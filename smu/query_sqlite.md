# query_sqlite

The command line tool for accessing the SMU databases is `query_sqlite.py`

The page describes how to use it, the various options, and gives examples.

If you have not followed [the installation instructions and gotten
the data](../README.md), do that first.

The sections below decribe the flags you can provide.


## Input

* `--input_sqlite`
Specifies the sqlite file to read (either the complete or standard), e.g.
`--input_sqlite 20220128_complete.sqlite`


## Output

* `--output_format`
There are a variety of different output formats
    * `pbtxt`: Human and machine readable test format that directly corresponds to the dataset.proto
    * `atomic2_input`: Generates the input files to go into the ATOMIC-2 fortran programs for computing ATOMIC-2 energies
    * `dat`:  Original .dat format generated by the FORTRAN code
    * sdf: There are three different options for SDF output controlling which geometries (initial and optimized) are included:  `sdf_opt`, `sdf_init`, `sdf_init_opt`.

* `--which_topologies` For sdf and `atomic2_input` output formats, this selects which topologies will be included as outputs
    * `all` : All matched topologies (from any of the three sets of bond lengths, see the other options below)
    * `best` : A single best matched topology based on the ITC criteria
    * `starting` : A single topology that was used during the initial calculations
    * `itc` : All matched topologies using the bond lengths used for original topology creation
    * `mlcr` : All matched topologies using the bond lengths based on covalent radii, following [Meng et al.](http://dx.doi.org/10.1002/jcc.540120716)
    * `csd` : All matched topologies using the bond lengths based from the Cambridge Structural Database compiled by [Allen et al.](http://dx.doi.org/10.1039/P298700000S1)

* `--output_path`
Where to write the output (uses stdout if not given)


## Selecting Molecules

The most straightforward way to get some molecules out is with
* `--random_fraction` where each record in the database is returned with this probability. Note that the entire database is read, so this can be quite slow.

During the build of the database, we add several indices for fast lookup. You can provide multiple options and you will get output for all of them.

* `--btids`: List of bond topology ids (e.g 85532). All molecules with this bond topology ID (including the detected bond topologies) are returned.

* `--mids`: List of molecule id (eg 85532001). Zero or one molecules will be returned for each mid.

* `--smiles`: List of SMILES strings (e.g. CC=O). All molecules with bond topologies represented by these SMILES (including the detected bond topologies) are returned. Note that these inputs will be recanonicalized using the same procedure we used to build the database so alternative SMILES that are the same bond topology will return the same set of results. Note that if aromatic smiles are given one of the kekulized forms will be chosen somewhat arbitrarily. However, because of our geometry detection, this should still return the set of molecules covering the other kekulized form.

* `-–stoichiometries`: List of stoichiometries like (C6H6) to query. Case does not matter.

* `--smarts`: Use a [SMARTS pattern](https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html) to select a set of molecules. This effectively finds a set of bond topology ids and then returns all molecules for those bond topology ids (like `--btids`. A couple points to note
    * Unlike the other options here, only a single SMARTS pattern is allowed.
    * Since we do not use any notion of aromatic bonds, you should only use aliphatic atoms (i.e. 'C' and not 'c') in your SMARTS.

## Geometry Based Searching
To support geometric manipulations, a file generated from our pipeline with histograms of observed bond lengths is needed to provide the default bond length information. This file is called `20220128_bond_lengths.csv`

* `--bond_lengths_csv` : should be given the path of the above file.

* `--bond_lengths`
To support specified alternative bond length restrictions, you can specify lengths on the command line. This is a comma separated list of terms
    ```
    <atom1><bond char><atom2>:<mindist>-<maxdist>
    ```
    Where
    * `<atom1>`, `<atom2>`: are characters in “CNOF*” (not that hydrogen is not allowed as we always attached hydrogens to their nearest heavy atom. The order is irrelevant. ‘*’ means to match any of CNOF
    * `<bond_char>`: is one of “-”, “=”, “#”, “.”, “\~”. These are the bond specifiers in SMILES strings, with ‘\~’ meaning any bond.
    * `<mindist>`, `<maxdist>`: lengths in angstroms. Either of them can be left off. For mindist, this is equivalent to 0. For maxdist, we use a “right_tail_mass” with an exponential decay of the pdf (meaning that a small amount of probablity mass is given to all lengths to infinity)

    For example, these are all valid terms
    ```
    N-N:1.3-1.7
    N=O:1.3-1.5
    C#C:1.21212-1.3456
    C.O:2.0-9999
    N~N:1.3-1.9    (which means all NN bonds are between 1.3 and 1.9)
    ```

    Note that `--bond_lengths` will override any data for that pair in `--bond_lengths_csv`

    Implementation note of interest: `--bond_lengths_csv` creates a EmpiricalLengthDistribution for each atom type pair. `--bond_lengths` creates a FixedWindowLengthDistribution for each given pair to replace the EmpiricalLengthDistribution

* `--topology_query_smiles` To do a geometry based search, you then provide a comma separated list of SMILES strings specifying the geometry to search. Note that if `--bond_lengths` are not given, this will return the same molecules as `--smiles`.

### How does this work internally?
Each molecule is associated with a hydrogen specific stoichiometry. That is, the number of hydrogens bonded to a heavy atom is part of the atom type. So a carbon with no hydrogens is “c” but with two hydrogen is “ch2”. Examples:
* benzene: (ch)6
* water: (oh2)   (note that the 1 is implicit)
* ethylene: (ch2)2
* acrylic acid: (c)(ch)(ch2)(o)(oh)

In order for a molecule to match a query geometry, the hydrogen specific stoichiometry must match. So we locate all the molecules with matching hydrogen specific stoichiometry. For each one, we run the geometry detection algorithm (with the modified bond lengths from `--bond_lengths`). If one of the detected bond topologies is the same as the original geometry given in `--topology_query_smiles`, then the molecule is returned.

Note that this is not an especially efficient algorithm, but it’s able to reuse exactly the same geometry detection code so that we ensure everything is consistent.


## Redetecting Bond Topologies

Similar to the [geometry based searching](#geometry-based-searching), you can use the modification of allowed bond lengths (via the `--bond_lengths_csv` and `--bond_lengths` arguments) to perform topology sensing on any molecule returned (i.e. from the options in [Simple Indexed Selection](#simple-indexed-selection) )

* `--redetect_topology` Just specify this flag (in addition to `--bond_lengths_csv`, `--bond_lengths`)


## Examples
Find the record for a specific molecule id.
```
python -m smu.query_sqlite \
--input_sqlite 20220128_complete.sqlite \
--mids 85485001 \
--output_format sdf_opt
```

Find the records for a bunch of molecule ids and send the output to a file
```
python -m smu.query_sqlite \
--input_sqlite 20220128_complete.sqlite \
--mids 1001,4001,1193001,81680048,91856010,102959011,102959027,108993002,200252001,405360002,899649001,899650001,899651001,899652001  \
--output_path /tmp/example.pbtxt
```

Find all the molecules for the given bond topology ids. Note that this searches by all detected topologies, not just the topology that originally created the molecule.
```
python -m smu.query_sqlite \
--input_sqlite 20220128_complete.sqlite \
--btids '581948,532611,532626,540263'
```

Find all the molecules corresponding to a given SMILES and output to sdf. Note that this internally canonicalizes the SMILES so that both the forms given here return the same thing.
```
python -m smu.query_sqlite \
--input_sqlite 20220128_complete.sqlite \
--smiles 'NN=O' \
--output_format sdf_opt \
--output_path /tmp/NNO_v0.sdf

python -m smu.query_sqlite \
--input_sqlite 20220128_complete.sqlite \
--smiles 'O=NN' \
--output_format sdf_opt \
--output_path /tmp/NNO_v1.sdf
```

A geometry based query. A SMILES string is given to define the topology, but the valid length of all NN bonds is set to less than 2A.
```
python -m smu.query_sqlite \
--input_sqlite 20220128_complete.sqlite \
--output_format sdf_opt  \
--bond_lengths_csv 20220128_bond_lengths.csv \
--topology_query_smiles 'O=[N+]=NNN([O-])F' \
--bond_lengths 'N~N:-2.0'
```

A query by stoichiometry where we only return one bond topology for each molecule.
```
python -m smu.query_sqlite \
--input_sqlite 20220128_standard.sqlite \
--stoichiometries c6h14 \
--output_format sdf_opt \
--sdf_include_all_bond_topologies=False \
--output_path /tmp/c6h14.sdf
```