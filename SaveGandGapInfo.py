import numpy as np
import pymatgen
import pandas as pd
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
import warnings
import pickle
import glob
import sys
import os
warnings.filterwarnings("ignore", message=".*Pauling electronegativity for.*")
warnings.filterwarnings("ignore", message=".*ssues encountered while parsing CIF:.*")
pd.options.mode.copy_on_write = True 

my_API_key = 'N9VCGSyOzfB7eSuZDgZ1OWONAFyv9ZXP'
mp_139K_dir = '/Users/chimamcgruder/Work_General/ClimateBase/Materials/crystal_untagged_800K-main/dataset/mp_139K/'
MainPath = '/Users/chimamcgruder/Work_General/ClimateBase/Materials/MLtests/'

def RestructureDF(df): 
	"""To get the crystal labels and add them as the "row labels." 
	This will ultimately be set as another column when reopened using df.read_csv() WITHOUT index_col=[0], 
	but I want to save it here as a row label to optimize space. """
	row_labels = df.iloc[:, 0].values  # Extract row labels as a NumPy array
	df.drop(columns='Unnamed: 0', inplace=True)  # Drop the 'Unnamed: 0' column. This is the column of row crystal labels
	df = pd.DataFrame(df.values, index=row_labels, columns=df.columns)  # Recreate DataFrame with new index and existing columns
	return df

def SaveBandGapInfo(CSVfile, group, ToSave, cif_path=mp_139K_dir, matgenKEY=my_API_key): 
	"""made a seperate function to get the band gap info cause it takes so freaking long, I need to split this part up in chuncks
	group = to split the data from the csv file into groups. So can save the band gap data from that group seperately. So I don't have to run everything in one go
			Need to be a 2D list ([A, B]) where the first element is how many chuncks you want and the 2nd element is which chunck this is supposed to be B i.e. a # between 1 and A
	"""
	if '/' in ToSave: #then there is path information in the name
		path = ToSave[:-len(ToSave.split('/')[-1])] #get only the path info 
		os.makedirs(path, exist_ok=True) #make any directories that aren't already created

	df = pd.read_csv(CSVfile) #not sure if the order of the rows will be maintained... need to make sure that'sso
	labels = np.array(df.iloc[:, 0])
	extras = len(labels)%group[0] # to make get an remainders when that would arise from dividing into group
	splits = np.split(labels[:-extras], group[0]) #remove remainder here because it has to be split in even groups
	splits[-1] = np.append(splits[-1], labels[-extras:])#add the remainder to the last split

	######## To Fix Chima's stupid mistake ########
	df_BD_63per = pd.read_csv(MainPath+'mp_139K_11feat/63perc_feature_BandGap_DATA.csv') #load the BD data I've already saved
	CryLab63per = df_BD_63per.iloc[:, 0]
	CIFs63per = []
	for cl63 in CryLab63per: #to get just the .cif file names for all data in '63perc_feature_BandGap_DATA.csv'
		CIFs63per.append(cl63.split('_')[1])
	CryBG_63per_df = df_BD_63per.iloc[:, -1] #to get just the Band Gap info
	CryBG_63per_df = pd.concat([CryLab63per, CryBG_63per_df], axis=1) #to add the extra column for full crystal labeling
	CryBG_63per_df.index = CIFs63per #to relabel the index of each row with the .cif file name
	######## To Fix Chima's stupid mistake ########

	MissingBG, BANDGAPs = [], []#to keep track of the crystals that band gap info can't be found on. And the band gap info in eV
	CrysLab, band_gap = [], -1e10 #here define band_gap as an unreasonable number, so the variable is defined, but we if it somehow gets accidently puts as a crystals actual band_gap value, I'll know 
	crys2extract = splits[group[1]]
	for crys in crys2extract:
		CIF = crys.split(':')[-1]

		######## To Fix Chima's stupid mistake ########
		if CIF in CIFs63per: #if band gap info is already saved, then I don't have to search through the materials project... again
			print ('\t\t\t found band gap info for', CIF, "in '63perc_feature_BandGap_DATA.csv'")
			print ('\t\t\t\t Band Gap:', CryBG_63per_df.at[CIF, 'band_gap'])
			print ('\t\t\t\t fixing my mistake...')
			band_gap = float(CryBG_63per_df.at[CIF, 'band_gap'])
			BANDGAPs.append(band_gap)
			CrysLab.append(CryBG_63per_df.at[CIF, 'Unnamed: 0'])
		else: ######## To Fix Chima's stupid mistake ########
			structure = Structure.from_file(cif_path+CIF) # Load the CIF file
			with MPRester(matgenKEY) as mpr:
				try:
					results = mpr.find_structure(structure) # Find materials matching the structure
					print ('Materials Project ID: '+str(results))
					material_data = mpr.materials.summary.search(material_ids=[results], fields=["band_gap"])
					if len(material_data) == 0:
						print (f"WARNING!!! Can't find the Band Gap of {structure.composition}({CIF},{results}) in the materials project!!!")
						MissingBG.append(CIF+'_'+str(results))
					else:
						band_gap = float(material_data[0].band_gap)
						print(f"Band gap for {structure.composition}({CIF}): {band_gap:.4f} eV") # Print out the results
						CIF +='_'+results #to include the MP idea as part of the crystal ID
						CrysLab.append(CIF), BANDGAPs.append(material_data[0].band_gap)
				except:
					MissingBG.append(CIF+'_NotFoundInMP')
					print('WARNING!!! Was unable to find a molecule in the materials project with the same structure and composisiton matching '+CIF+'('+str(structure.composition)+')') #WARNING!!! Was unable to find a molecule in the materials project with the same structure and composisiton matching 66079.cif(Ga2 Cu4)
		if len(CrysLab) != len(BANDGAPs): # still haven't figured out what causes this error.... Think it's because it i's crashing in the else statement (line 77) before getting to CrysLab.append(CIF), but after BANDGAPs.append()
			print ("ERROR!!!\n Crystal length is ",len(CrysLab), "but band gap length is", str(len(BANDGAPs))+'!!!')
			print ("Last 3 entries in CrysLab:", CrysLab[-3:], "Last 3 entries in BANDGAPs:", BANDGAPs[-3:])
			sys.exit("ERROR!!!")
		if type(band_gap) != float: # still haven't figured out what causes this error.... Think it's because it i's crashing in the else statement (line 77) before getting to CrysLab.append(CIF), but after BANDGAPs.append()
			print ("ERROR!!!\n type of band_gap is", type(band_gap), str(band_gap)+'!!!')
			print ("Last 3 entries in CrysLab:", CrysLab[-3:], "Last 3 entries in BANDGAPs:", BANDGAPs[-3:])
			sys.exit("ERROR!!!")	
	BANDGAPs = np.array(BANDGAPs)
	if -1e10 in BANDGAPs: #to make sure the bandgap initialization was never used
		glitches = np.where(BANDGAPs == -1e10)[0] #find where fake BANDGAPS are
		glitches = np.array(CrysLab)[glitches] #find corresponding crystal labels
		print ("ERROR!!!\nThe following Crystals were given fake bandgap values:")
		for gl in glitches:
			print (gl)
		sys.exit("ERROR!!!")
	df = pd.DataFrame(BANDGAPs, index=pd.Index(CrysLab), columns=pd.Index(['band_gap']))
	print ("Final shape of pandas DataFrame:", df.shape)
	df.iloc[:, 0].values
	df.to_csv(ToSave+'_BandGaps.csv')
	with open(ToSave+'_missingGaps.pkl', 'wb') as pkl:
		pickle.dump(MissingBG, pkl)		
	return None

def CombineBandGap_with_features(BGfiles, misssingBGfiles, FeatureFile, FinalFile):
	""" To combine the band gap info and feature data into one pandas data frame
	BGfile = base name of the file where all the band gap info is stored. 
			Since there's multiple files with the same name, this will be a 2D list where the first entery is everything before the counting and the 2nd entery is everything after
	"""
	if '/' in FinalFile: #then there is path information in the name
		path = FinalFile[:-len(FinalFile.split('/')[-1])] #get only the path info 
		os.makedirs(path, exist_ok=True) #make any directories that aren't already created

	BG_Fs_list = glob.glob(BGfiles[0]+'*'+BGfiles[1]) #to get all of the current relevent BG files
	featured_df = pd.read_csv(FeatureFile)
	print ("featured_df initial shape:", featured_df.shape)
	new_columns_df = pd.DataFrame(columns=['band_gap'])
	featured_df = pd.concat([featured_df, new_columns_df], axis=1) #to add the extra column for band gap
	print ("featured_df shape after adding empty BG column:", featured_df.shape)
	df_labels = list(featured_df.iloc[:, 0])
	
	# to get a list of .cif files corresponding to all crystals found in the main featured_df 
	cif_list = [] 
	for lab in df_labels:
		cif_list.append(lab.split(':')[-1])
	
	rows2drop = []
	#To add the relevent band gap info
	for i, BGf in enumerate(BG_Fs_list):#Here I assume that the files are a number from 0 to len(BG_Fs_list)-1 without missing files and the same ordering is for the Misses_list
		df_BGi = pd.read_csv(BGf)
		df_BGi_labels = list(df_BGi.iloc[:, 0]) #get the names of the specific crystals for this cluster of band gap info as a list 
		for r, BG in enumerate(df_BGi_labels): #to iterate through each crystal
			cifTAG, molNm = BG.split('_')[1], ''
			if '.cif' not in cifTAG: #because I had 2 seperate ways of storing the crystal labels, the .cif tag is different half the time
				cifTAG = BG.split('_')[0]
				molNm = featured_df.at[IndxInMain_DF, 'Unnamed: 0'].split('_')[0]+'_'
			IndxInMain_DF = cif_list.index(cifTAG) #get the index of the specific crystal in terms of the main dataframe
			featured_df.at[IndxInMain_DF, 'band_gap'] = df_BGi.at[r, 'band_gap'] #fill in the main dataframe with the band gap info
			if cifTAG != featured_df.iloc[IndxInMain_DF,0].split(':')[-1]: #check to make sure I'm editing information that corresponds to the same crystal in both featured_df and df_BGi
				sys.exit("ERROR!!! Crystal .cif label of the Band gap info source is '"+cifTAG+"' But the main dataframe says you're working with .cif crystal '"+featured_df.iloc[IndxInMain_DF,0].split(':')[-1]+"'!!!")
			featured_df.at[IndxInMain_DF, 'Unnamed: 0'] = molNm+df_BGi.at[r, 'Unnamed: 0'] #to replace the crystal composition and .cif tag with the crystal composition, .cif tag AND materials project ID

		#To get the info about crystals missing band gap data
		with open(misssingBGfiles[0]+str(i)+misssingBGfiles[1], 'rb') as f:
			missingBG = pickle.load(f, encoding="latin1")
		for M in missingBG:
			bad_cif = M.split('_')[0]
			rows2drop.append(cif_list.index(bad_cif))
	
	#remove the missing band gap data at the end, so not to mess up the indexing		
	if len(rows2drop) > 0:
		featured_df = featured_df.drop(rows2drop)
	print ('dropped rows length:', len(rows2drop))
	print ("featured_df shape after removing all missing BG info:", featured_df.shape)

	featured_df = RestructureDF(featured_df) #To set the first column (crystal name) as the row label before saving the df

	featured_df = featured_df.dropna(subset=['band_gap'])
	print ("featured_df shape after removing all NaNs (i.e. data that I haven't gotten BG info for yet) and reshaping so crystal label is row axis and NOT a column:", featured_df.shape)
	print (featured_df[:-10])
	featured_df.to_csv(FinalFile)
	return None

def MakeTrainingTestCrossData(DF, train_size, testFrac=.25, crossValSize=10,basePath=MainPath, extraFolderNaming=''): 
	"""" To first make the Test data (which will be seperate from everything). Then make the training data, which you'll split into 'crossValSize' smaller subgroups for cross validation 
		 For crossValSize set to 10 or smaller training size and increase it up to 3 so we can consider larger datasets
	train_size = regardless of how large the dataset is, you'll want to specify the size of your training sample, because we are trying to test how well the model works with different data sizes
						since we are doing cross validation, we want to mulitple the train_size by 'crossValSize' so we'll have 'crossValSize' equally lengthed training data. Thus our total subdata size is total_subdata_size = crossValSize*train_size + testFrac*train_size
						This means our max train_size is  = DF.shape[0]/(crossValSize*+testFrac)
						with DF.shape[0], crossValSize, testFrac = 80K, 3, .25 max(train_size) = 21.333K | with DF.shape[0], crossValSize, testFrac = 80K, 5, .25 max(train_size) = 12.8K | with DF.shape[0],crossValSize, testFrac = 80K, 10, .25 max(train_size) = 6400
						use [1000, 3000, 6000] with crossValSize = 10 && [8000, 10000, 14000] crossValSize = 5 && [17000, 20000, 24000] crossValSize = 3 && [28000, 30000, 35000] crossValSize = 2
	testFrac = the fractional size of the test data 
	"""
	ALLdata = pd.read_csv(DF)
	if train_size > int(np.floor(ALLdata.shape[0]/(crossValSize+testFrac))): 
		sys.exit("train_size, ALLdata.shape[0], crossValSize = "+str(train_size)+", "+str(ALLdata.shape[0])+", "+str(crossValSize)+" and can't satify everything. \n i.e. Get more data, or make train_size or crossValSize smaller.")
	TotalDataSize = int(round((train_size*crossValSize) + (train_size*testFrac))) # total data set includes training and test data (with cross val data)
	
	#### To randomly pull indeces representing specific rows in the main dataframe to generate the required data subset 
	ALLdata_indeces = np.arange(ALLdata.shape[0])
	SubDataIndces = np.random.choice(ALLdata_indeces, size=TotalDataSize, replace=False)
	test_size = int(round(testFrac*train_size))
	if test_size%crossValSize == 0: #if test size isn't a whole number when multiplying by the testFrac
		test_size += test_size%crossValSize #then add the remainders to the test set, so there will always be an even number of training sets for cross validation
	train_size = int((TotalDataSize - test_size)/crossValSize)
	print ("ALLdata.shape:",ALLdata.shape, "np.max(ALLdata_indeces):",np.max(ALLdata_indeces))
	print ("train_size and test_size:", train_size, test_size, "true percent of test_size (relative to train_size):", str(round((test_size/train_size)*100,3))+'%\n' )

	####To get a list of indeces for all the cross val training sets and the test set
	#create a directory to store all of subdataframes
	if basePath[-1] != '/': #make sure the basepath name has a backslash to end title of path
		basePath+='/'
	newpath = basePath+'DataSubset_'+extraFolderNaming+str(train_size)+'training_'+str(crossValSize)+'crossVal'
	os.makedirs(newpath, exist_ok=True)  #make any directories that aren't already created
	
	np.random.shuffle(SubDataIndces)#just to introduce a little more randomness
	for cV in range(crossValSize):
		Rand_train_indcs_i = list(SubDataIndces[cV*train_size:(cV+1)*train_size])
		train_df_i = ALLdata.iloc[Rand_train_indcs_i, :]
		train_df_i = RestructureDF(train_df_i) #To set the first column (crystal name) as the row label before saving the df
		print ("train_df_i.shape:", train_df_i.shape, "max(Rand_indx):", np.max(Rand_train_indcs_i), "min(Rand_indx):", np.min(Rand_train_indcs_i))
		train_df_i.to_csv(newpath+'/trainingDat_sub'+str(cV)+'.csv')
	Rand_test_indcs_i = list(SubDataIndces[(cV+1)*train_size:])
	test_df = ALLdata.iloc[Rand_test_indcs_i, :]
	test_df = RestructureDF(test_df) #To set the first column (crystal name) as the row label before saving the df
	print ("test_df.shape:", train_df_i.shape, "np.max(Rand_test_indcs_i):", np.max(Rand_test_indcs_i), "np.min(Rand_test_indcs_i):", np.min(Rand_test_indcs_i))
	test_df.to_csv(newpath+'/testDat.csv')
	return None

def RemoveAllConductors(fullCSV): #conductors have a bandgap of 0. Crystals with 0 band gap seem hard for the RF to predict
	df = pd.read_csv(fullCSV, index_col=[0]) 
	print ('dataframe initial shape:', df.shape)
	df = df[df['band_gap'] > 1e-4]
	print(np.min(df.iloc[:,-1].values))
	BasName = fullCSV.split('/')[-1]
	path = fullCSV[:-len(BasName)] #save the data in the same path as the orginal .csv files
	name = BasName.split('.csv')[0] #and the same name, just add note that I removed 0s
	df.to_csv(path+'/'+name+'_NOconductors.csv')
	print ('dataframe final shape:', df.shape)
	return None

if __name__ == "__main__":
	csv_file = MainPath+'mp_139K_11feat/featured_DATA.csv'
	save_path = MainPath+'mp_139K_11feat/BandGapDat/'
	# iters = 100
	# completedJs = 89
	# for i in range(iters-completedJs):
	# 	print ('Working on group', i+completedJs)
	# 	SaveBandGapInfo(csv_file, [iters,i+completedJs], save_path+'Iter'+str(i+completedJs), cif_path=mp_139K_dir, matgenKEY=my_API_key)
	# 	print ('\n\n\n')


	BG_fil_format = [save_path+'Iter', '_BandGaps.csv']
	MIAs_fil_format = [save_path+'Iter', '_missingGaps.pkl']
	csv_final_file = MainPath+'mp_139K_11feat/Feature_BandGap_DATA.csv'
	# CombineBandGap_with_features(BG_fil_format, MIAs_fil_format, csv_file, csv_final_file)
	# RemoveAllConductors(csv_final_file)
	csv_final_fileNOconducts = MainPath+'mp_139K_11feat/Feature_BandGap_DATA_NOconductors_LatticMatrix.csv'
	MakeTrainingTestCrossData(csv_final_fileNOconducts, 3000, .25, basePath=MainPath+'mp_139K_11feat/', extraFolderNaming='NOconductors_LatticMatrix_')

