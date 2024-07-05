import numpy as np
import pymatgen
import glob
import pandas as pd
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.ext.matproj import MPRester
import sys
import os
import warnings
import pickle
warnings.filterwarnings("ignore", message=".*Pauling electronegativity for.*")
warnings.filterwarnings("ignore", message=".*data available for superconduction_temperature for.*")
warnings.filterwarnings("ignore", message=".*ssues encountered while parsing CIF:.*")
warnings.filterwarnings("ignore", message=".*o data available for bulk_modulus for.*")

"""NOTES: 
1) Not all elements have an empirical measurement of their atomic radius, for those elements without empirical measurements, I just used the experiementally measured radius
2) He, Ne, and Ar don't have elecronegativity measurements because they don't naturally bond with other atoms, makting the measurements impossible. 
	Thus, for those elements I'll just omit that atom in the contribution to the avg electronegativity. I'll ignore the few crystals that are just composed of noble gases
3) Only a few elements have superconductivity values under normal conditions. Thus, only include the atoms with such values in the average calculations
		... Still tryna think about how to handle crystals that don't have any atoms with superconductivity in it at all... FOR NOW just ignore those crystals (42% of data)
4) The bulk modulus can't be obtained for Gases, liquids, and weakly interacting molecular crystals. Thus, pytmatgen can't calculate the bulk modulus for O,H,N,F,Ge,In,Os,Sr,Zr,Ga,Tc,Xe,Pa,Pu,Ac,Np,He,Ar,Kr,Ne
	For those elements I'll just omit that atom in the contribution to the avg electronegativity. I'll ignore the few crystals that are just composed of noble gases

TODOs: 
1) Check out pymatgen calculates these values, because a lot of them depend on specific conditions. It might not matter, so long as I'm consistent, but I'd at least want to know the discrepencies between one value from the next.
2) use mean, max, & std in distances matrix between atoms (to include some information about the actual structure of the atom)
					#maybe also lattice parameters, see https://www.youtube.com/watch?v=14CHTYQImeQ (35min deep) and https://colab.research.google.com/drive/1p6ORktVuni6dhGLB70WM_PLbdPvrDh_N#scrollTo=rvw2H0R9fCKi
					#also include total number of atoms
					#fractional number of atoms used to calculate each feature
					#band gap of each individual atom
"""

mp_139K_dir = '/Users/chimamcgruder/Work_General/ClimateBase/Materials/crystal_untagged_800K-main/dataset/mp_139K/'
crystals = glob.glob(mp_139K_dir+'*.cif') #load all of the data
print ('total number of .cif files found:', len(crystals))
my_API_key = 'N9VCGSyOzfB7eSuZDgZ1OWONAFyv9ZXP'
dirPath = '/Users/chimamcgruder/Work_General/ClimateBase/Materials/MLtests/'

def CalAvgProp(Comp, feature, ID, NoFeaturesDir, Name=None, Print=True, SavePrints=None): 
	"""to get the average specified 'feature' of a given crystal feature needs to 
	be a string of the specific label (e.g. electronegativity = X, atomic radius = atomic_radius)"""
	if Name:
		str_nam = Name
	else:
		str_nam = feature
	element_cnts = Comp.get_el_amt_dict()
	elements = list(element_cnts.keys())
	TotQuant, AtmCnt = 0, 0
	for e in elements:
		molecule = Element(e)
		if feature == 'boiling_point' and e == 'Pa': #the boiling point of Pa isn't well measured, but we'll use Google's best guess
			Property = 4273 #[K] https://www.rsc.org/periodic-table/element/91/protactinium
		elif feature == 'atomic_radius': 
			Property = molecule.atomic_radius #the empirical radius is not available for all elements. 
			if not Property: #in those cases just used the calcualted radius
				if Print:
					print ('Using the calculated radius for', e)
					if SavePrints:
						SavePrints.write('Using the calculated radius for '+str(e)+'\n')
					Property = molecule.atomic_radius_calculated
		elif feature == 'mineral_hardness' and e == 'C':#since carbon can be in graphite and diamond form there are 2 valid values
			Property = 0.5 #Let's just use that of Carbon graphite for now		
		elif feature == 'ionization_energies': #we'll only use the first ionization energy
			Property = molecule.ionization_energies[0]
			str_nam = 'first_'+feature
		else:
			Property = getattr(molecule, feature) #detail about getattr: It's just to get an unspecified attribute of an object: https://www.programiz.com/python-programming/methods/built-in/getattr
		if not Property or np.isnan(Property): # if for whatever reason the specific property doesn't exists for that atom, skip it. NaN counts as values is missing!				
			if Print and feature != 'superconduction_temperature': #only supposed to have a few elements that actually have superconductivity. So no need to print warning	
				print ('\nPASS AUF!!!! No '+str_nam+' found for '+e+'!!!') #when the value is 0, assume it doesn't have a value and don't include it
				print ('\tIn crystal('+str(ID)+'):', Comp, '\n')
				if SavePrints:
					SavePrints.write('\nPASS AUF!!!! No '+str_nam+' found for '+e+'!!!'+'\n')
					SavePrints.write('\tIn crystal('+str(ID)+'): '+str(Comp)+'\n\n')
			# sys.exit()
			continue
		try:
			TotQuant += Property*element_cnts[e]
		except:
			print ('ERROR!!! \n crystal('+str(ID)+'):',Comp, 'feature:',feature, 'Property:', Property, 'element:', e, 'element_cnts[e]:', element_cnts[e])
			print ('Property type:', type(Property), '\n ERROR!!!')
			if SavePrints:
				SavePrints.close()
			sys.exit()
		AtmCnt += element_cnts[e]
		if Print:
			print(f'{str_nam} of {e}: {Property}')
			if SavePrints:
				SavePrints.write(f'{str_nam} of {e}: {Property}\n')
	if not AtmCnt: #The property for each element in the molecule don't have a value
		NoFeaturesDir[feature].append(ID)
		if Print:
			print("DON'T HAVE THE AVERAGE \033[1m",str_nam,"\033[0m FOR THIS CRYSTAL("+ID+"):"+str(Comp)) 
			if SavePrints:
				SavePrints.write("DON'T HAVE THE AVERAGE "+str_nam+" FOR THIS CRYSTAL("+ID+"):"+str(Comp)+'\n')
		return NoFeaturesDir
	if Print:
		print ('avg_'+str_nam+':', TotQuant/AtmCnt, 'out of',AtmCnt,'total atoms')
		if SavePrints:
			SavePrints.write('avg_'+str_nam+': '+str(TotQuant/AtmCnt)+' out of '+str(AtmCnt)+' total atoms\n')
	return TotQuant/AtmCnt




def SaveFeatures(features, BatchSize, BatchName, ToUpdate=False, CrysList=crystals, MandatoryUpdates=100, matgenKEY=my_API_key, GetBandGap=True):
	"""To store the 19 (need to add cation electronegativity later) features for each molecule in a pandas array. This could be easily extracted from pymatgen directly, but want to save time in training. 
	So store all information directly. This is also the same logic why I'm not just storing the information per element in a pandas, 
	then just averaging for a given molecule in the training. I'd rather save as much run time by having the data stored right there """
	
	if '/' in BatchName: #then there is path information in the name
		path = BatchName[:-len(BatchName.split('/')[-1])] #get only the path info 
		os.makedirs(path, exist_ok=True) #make any directories that aren't already created

	MissingFeatures = {} #to keep track of which features were missing for a specific molecule
	if GetBandGap: #might want to turn this off to save time when I'm checking what's wrong with other features
		Bfeatures = features+['band_gap'] #add band gap as a feature, though it's really the value we are trying to learn
	else:
		Bfeatures = features
	for f in Bfeatures:
		MissingFeatures[f] = []
	if BatchSize == 'all':
		TestCrystals = CrysList
	else:
		TestCrystals = np.random.choice(CrysList, size=int(BatchSize), replace=False)
	TxtUpdates = open(BatchName+'_prints.txt', 'w+')
	Print_txt = 'total number of crystals using: '+str(len(TestCrystals))
	print (Print_txt), TxtUpdates.write(Print_txt+'\n')
	MolName, BAND_GAPs = [], []
	MandUps, Nxt = int(len(TestCrystals)/MandatoryUpdates), 0 #gotta give at least a few updates
	for ci, C in enumerate(TestCrystals):
		structure = Structure.from_file(C) # Load the CIF file
		SKIPPED = False
		# Get the Composition
		Composition = structure.composition
		CrysID = TestCrystals[ci].split('/')[-1]
		if ToUpdate:
			Print_txt = 'Crystal ('+CrysID.split('.cif')[0]+'): '+str(Composition)+' #'+str(ci)
			print (Print_txt), TxtUpdates.write(Print_txt+'\n')
		else:
			if ci == Nxt: #to provide mandatory (regardless of if ToUpdate is True) updates of how much of the data has been saved
				print (str(round((ci/len(TestCrystals))*100, 2))+"% complete")
				print ('Crystal ('+CrysID.split('.cif')[0]+'):', Composition, '#'+str(ci), '\n') #don't need to save the progress update in the .txt file
				Nxt+=MandUps
		PD_data_featD = np.zeros(len(Bfeatures)) 
		for fi, F in enumerate(features): #here use features and not Bfeatures, because band gap isn't actually a pymatgen feature and obtained differently
			if ToUpdate:
				print (ci, fi), TxtUpdates.write(str(ci)+' '+str(fi)+'\n')
				Prt = True
			else:
				Prt =False
			feat_ci_fi = CalAvgProp(Composition, F, CrysID, MissingFeatures, Print=Prt, SavePrints=TxtUpdates)
			if type(feat_ci_fi) != float: #As soon as we realize we can't add one feature for this specific crystal, exit inner for loop, because we aren't adding this crystal to our final pandas dataframe
				MissingFeatures = feat_ci_fi #still update which specific crystal isn't going to be included in the final dataframe
				SKIPPED = True
				break 
			PD_data_featD[fi] = feat_ci_fi
		if GetBandGap:
			if not SKIPPED: 
				#### After making sure that the features can be obtained, now try to get the band gap info. Do this last cause it takes the most time 
				with MPRester(matgenKEY) as mpr:
					try:
						results = mpr.find_structure(structure) # Find materials matching the structure
						if ToUpdate:
							Str2Prnt = 'Materials Project ID: '+str(results)
							print (Str2Prnt), TxtUpdates.write(results+'\n')
						material_data = mpr.materials.summary.search(material_ids=[results], fields=["band_gap"])
						if len(material_data) == 0:
							SKIPPED = True
							if ToUpdate:
								print (f"WARNING!!! Can't find the Band Gap of {Composition}({CrysID},{results}) in the materials project!!!")
								TxtUpdates.write(f"WARNING!!! Can't find the Band Gap of {Composition}({CrysID},{results}) in the materials project!!!\n")
							MissingFeatures['band_gap'].append(CrysID+'_'+str(results))
						else:
							band_gap = material_data[0].band_gap
							PD_data_featD[fi+1] = band_gap
							CrysID +='_MPid:'+results #to include the MP idea as part of the crystal ID
							if ToUpdate:
								print(f"Band gap for {Composition}: {band_gap:.4f} eV") # Print out the results
								TxtUpdates.write(f"Band gap for {Composition}: {band_gap:.4f} eV\n")
					except:
						SKIPPED = True
						MissingFeatures['band_gap'].append(CrysID+'_NotFoundInMP')
						if ToUpdate:
							Str2Prnt = 'WARNING!!! Was unable to find a molecule in the materials project with the same structure and composisiton matching '+CrysID+'('+str(Composition)+')'
							print(Str2Prnt), TxtUpdates.write(Str2Prnt+'\n')
		if not SKIPPED: #don't add molecule name if we ended up skipping it
			MolName.append(str(Composition.reduced_formula)+'_ID:'+CrysID)
			try: 
				PD_data = np.vstack((PD_data, PD_data_featD))
			except: # if this is the first element in the array, then initate the array here
				PD_data = PD_data_featD
		if ToUpdate:
			print ('\n\n'), TxtUpdates.write('\n\n')
	print ('Number of crystals with all features:', len(MolName), '\nNumber of features per crystal:', len(Bfeatures))
	df = pd.DataFrame(PD_data, columns=pd.Index(Bfeatures), index=MolName) #if I save the dataframe like this, then reopen it without specifying index_col=[0] is the labels for each row
	df.to_csv(BatchName+'_DATA.csv') #then the dataframe will create an extra unnamed column of just the crystal labels. I still think I should open it without specifying index_col=[0]
	df = pd.read_csv(BatchName+'_DATA.csv') # as that's better for future data manipulation. and keeping the formate like this where that 1st column is unnamed saves space. As if I were to label it then it'd store the indeces of each row in the .csv file (wasting storage)
	print ("Final shape of pandas DataFrame:", df.shape)
	for misF in list(MissingFeatures.keys()):
		print ('\033[1m',len(MissingFeatures[misF]), '\033[0mcrystals were omitted from dataframe because nothing can be found on \033[1m'+misF+'\033[0m')
	TxtUpdates.close()
	with open(BatchName+'_MIAs.pkl', 'wb') as pkl:
		pickle.dump(MissingFeatures, pkl)
	return None

def SaveMatrix_and_LatticInfo(fullCSV):
	"""matrix and lattice paramters, so we have parameters that are unique to the specific crystal. 
	All other parameters could have the same value for a different crystal if it's the same atoms (but different arangement of atoms)"""
	df = pd.read_csv(fullCSV, index_col=[0])
	print ("Intial data shape:", df.shape)
	bnd_gp = df.iloc[:,-1] #want the band_gap info to still be last
	modified_df = df.iloc[:,:-1]
	#### add more columns for matrix and lattice parameters
	newParams = ['maxDist', 'meanDist', 'stdDist', 'minDist', 'a', 'b', 'c', 'alpah', 'beta', 'gamma']
	new_columns = {param: np.empty(modified_df.shape[0]) for param in newParams} # Create a dictionary of new columns with empty arrays
	modified_df = modified_df.assign(**new_columns) # Assign new columns to the DataFrame
	molecule = list(df.index[:])
	for m in molecule: #get the distance matrix and lattice information, then add it to the df
		CiF = m.split('_')[1]
		structure = Structure.from_file(mp_139K_dir+CiF)
		composition = structure.composition
		DisMatrx = structure.distance_matrix
		DisMatrx = DisMatrx.flatten()
		DisMatrx = DisMatrx[np.where(DisMatrx > 0)[0]] #don't want to include distance to itself
		if len(DisMatrx) == 0: #then this will just be one atom ... right?
			maxDist, meanDist, stdDist, minDist = 0, 0, 0, 0
		else:
			maxDist, meanDist, stdDist, minDist = np.max(DisMatrx), np.mean(DisMatrx), np.std(DisMatrx), np.min(DisMatrx)
		a_param, b_param, c_param = structure.lattice.abc[0], structure.lattice.abc[1], structure.lattice.abc[2]
		alpha_param, beta_param, gamma_param = structure.lattice.angles[0], structure.lattice.angles[1], structure.lattice.angles[2]
		newVals = [maxDist, meanDist, stdDist, minDist, a_param, b_param, c_param, alpha_param, beta_param, gamma_param]
		for i in range(len(newVals)):
			modified_df.at[m, newParams[i]] = newVals[i]
	modified_df = pd.concat([modified_df, bnd_gp], axis=1) #add the band_gap at the end again
	basName = fullCSV.split('.csv')[0] #get the file name aside from the .csv ending
	basName+='_LatticMatrix.csv'
	print ("Final data shape:", modified_df.shape)
	modified_df.to_csv(basName)
	return None

"""FeaturesComplete+FeaturesRest = all the features that https://arxiv.org/pdf/2301.03372 used to predict band gap using random forest. 
However, with FeaturesRest, there were some elements that pymatgen didn't know the value of. For now I'm skipping those features, but 
I might have to calculate that manually on my own later"""
if __name__ == "__main__":
	FeaturesComplete = ["superconduction_temperature", "X", "atomic_radius", "van_der_waals_radius", "mendeleev_no", "molar_volume",
				"thermal_conductivity", "boiling_point", "melting_point", "ionization_energies", "bulk_modulus"]

	FeaturesRest = ["electrical_resistivity", "critical_temperature", "brinell_hardness","vickers_hardness", 'cation electronegativity'
				"youngs_modulus",  "rigidity_modulus", "mineral_hardness", "density_of_solid", "critical_temperature"] 
				#TODO: use mean, max, & std in distances matrix between atoms
						#maybe also lattice parameters, see https://www.youtube.com/watch?v=14CHTYQImeQ (35min deep) and https://colab.research.google.com/drive/1p6ORktVuni6dhGLB70WM_PLbdPvrDh_N#scrollTo=rvw2H0R9fCKi
						#also include total number of atoms
	FeatureTest = ["bulk_modulus"]

	rand_crstals = crystals
	
	# np.random.seed(43) #seed so I get the same random numbers everytime
	# random_numbers = np.random.randint(0, len(crystals), 1000) # Generate 100 random numbers
	# rand_crys100 = np.array(crystals)[random_numbers]
	# SaveFeatures(FeaturesComplete, 'all', dirPath+'mp_139K_11feat/featured', ToUpdate=False, GetBandGap=False)#, CrysList=rand_crys100)
	csv_final_fileNOconducts = dirPath+'mp_139K_11feat/Feature_BandGap_DATA_NOconductors.csv'
	SaveMatrix_and_LatticInfo(csv_final_fileNOconducts)



