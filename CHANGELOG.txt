***********************
Changelog for Pyflation
***********************

* v0.2.1 16/02/2012 - Added Sphinx documentation in docs directory. Tidied up 
	docstrings throughout the project and split some of the README file into USAGE.txt.
	Documentation is available online at http://pyflation.ianhuston.net/docs. 
	
* v0.2 01/12/2011 - Major new release including multifield calculations for first order
	perturbations. Second order calculations are not yet posible for multiple fields. 
	Models now include the parameter "nfields" which specifies how many fields are used.
	This parameter is used by the potentials function and needs to be passed to it.
	New potentials have been included in the cmpotentials module, including two field
	hybrid potentials and an nflation potential which allows an arbitrary number of fields.

	The multifield functionality includes calculating quantum mode 
	matrix for first order modes. This allows full calculation of all cross terms in 
	the power spectrum of the perturbations. 
	Added multifield syntax for PhiModels and the subclasses BackgroundModel
	and FirstOrderModel. Note that the result of the first order model is now complex,
	the previous split real and imaginary perturbations have been joined together. 
	This will break compatibility with results created with the previous version.
	
* v0.1 04/03/2011 - Initial Release

