"""
Genetic algorithm to identify good location for services.

Score_matrix:
    0: Number of hospitals
    1: Average distance
    2: Maximum distance
    3: Maximum admissions to any one hospital
    4: Minimum admissions to any one hospital
    5: Proportion patients within target time/distance
    6: Proportion patients attending unit with target admission numbers
    7: Proportion of patients meeting distance and admissions target
    8: 90th percentile travel
    9: 95th percentile travel
    10: 99th percentile travel

    
Programme written based on objects. For object classes are used each with one instance:    
"""

# Import general libraries
import pandas as pd
import numpy as np
import random as rn
import os
import sys
import datetime
from scipy.spatial.distance import pdist


class GlobVariables():
    """Global variables"""

    def __init__(self):
        # Set general model parameters and create output folder if necessary
        self.pareto_scores_used = [0, 1, 2, 3, 4, 5, 6, 7, 9]
        self.output_location = '1_no_limitations'
        self.target_travel = 30  # target distance/time
        self.target_min_admissions = 6000  # target sustainable admissions
        self.maximum_generations = 300  # generations of genetic algorithm
        self.fix_hospital_status = False  # Allows hospitals to be forced open or closed
        self.initial_random_population_size = 10000
        self.minimum_population_size = 1000
        self.maximum_population_size = 20000
        self.mutation_rate = 0.005
        self.max_crossover_points = 3
        self.use_regions = False  # Limits choice of hospital to regions
        self.proportion_new_random_each_generation = 0.05

        # Create output folder if neeed
        self.check_output_folder_exists()
        return

    def check_output_folder_exists(self):
        """Create new folder if folder does not already exist"""
        if not os.path.exists(self.output_location):
            os.makedirs(self.output_location)
        return


class Data():
    """
    Data class loads and stores core data for location algorithm.
    """

    def __init__(self, use_regions, fix_hospital_status):
        """Initialise data class"""
        # Define attributes
        self.admissions = []
        self.admissions_index = []
        self.hospitals = []
        self.travel_matrix = []
        self.travel_matrix_LSOA_index = []
        self.travel_matrix_hopspital_index = []
        self.hospital_count = 0
        self.loaded_population = []
        self.regions_dictionary = {}

        # Load data
        self.load_data()

        # Identify regions if required
        if use_regions:
            self.identify_region()
            self.create_regions_dictionary(fix_hospital_status)
        return

    # Check loaded data indicies for hospitals and LSOAs match
    def check_loaded_data_indices_match(self):
        """Check hospitals and LSOAs macth in number and text"""

        if len(self.admissions_index) != len(self.travel_matrix.index):
            sys.exit("LSOA admissions different length from travel matrix")

        check_lsoa_match = (self.travel_matrix.index == self.admissions_index).mean()
        if not check_lsoa_match == 1:
            sys.exit("LSOA admission names do not match travel matrix")

        if len(self.hospitals) != len(list(self.travel_matrix)):
            sys.exit("Hospital list different length from travel matrix")

        check_hospital_match = (list(self.travel_matrix) ==
                                self.hospitals.index).mean()
        if not check_hospital_match == 1:
            sys.exit("Hospital list names do not match travel matrix")

        return

    def create_regions_dictionary(self, use_fixed_status):

        hospitals_region = self.hospitals[['index_#', 'region', 'Fixed']]
        for index, values in hospitals_region.iterrows():
            index_number, region, fix = values
            index_number = int(index_number)
            # If using fixed hospitals ignore those with fixed status of -1
            if not all((use_fixed_status, fix == -1)):
                if region in self.regions_dictionary:
                    self.regions_dictionary[region].append(index_number)
                else:
                    self.regions_dictionary[region] = [index_number]
        return

    def identify_region(self):

        use_admissions_region = True

        if use_admissions_region:
            self.admissions_region = self.admissions_with_index['region']
        else:
            # Allocate admissions region to closest possible (used) hospital region
            mask = self.hospitals['Fixed']
            mask = mask.values != -1
            mask = mask.flatten()
            open_hospitals = self.hospitals.loc[mask].index
            # Get available hospital postcodes
            masked_matrix = self.travel_matrix.loc[:, open_hospitals]
            closest_hospital_ID = np.argmin(masked_matrix.values, axis=1)
            closest_hospital_postcode = open_hospitals[list(closest_hospital_ID)]
            self.admissions_region = self.hospitals.loc[closest_hospital_postcode]['region']

        # Adjust travel matrix so out of region hospitals have infinite travel distances
        x = list(self.travel_matrix)  # list of hospitals in travel matrix
        matrix_region = list(self.hospitals.loc[x]['region'])  # list of regions of hospitals in travel matrix
        matrix_hospital_region = np.array([matrix_region, ] * len(self.admissions_region))  # repeat list of regions
        matrix_LSOA_region = np.repeat(self.admissions_region, self.hospitals.shape[0]).values.reshape(len(
            self.admissions_region), self.hospitals.shape[0])
        hospital_not_in_LSOA_region = matrix_LSOA_region != matrix_hospital_region
        matrix_correction = np.ones(self.travel_matrix.shape)
        matrix_correction[hospital_not_in_LSOA_region] = np.inf
        self.travel_matrix += matrix_correction
        return

    def load_data(self):
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), 'Loading data')
        # Load hospital list
        self.hospitals = pd.read_csv('data/hospitals.csv', index_col=0)
        self.hospital_count = len(self.hospitals)
        self.hospitals['index_#'] = list(range(0, self.hospital_count))

        # Load admissions and split index from data
        self.admissions_with_index = pd.read_csv('data/admissions.csv')
        self.admissions_index = self.admissions_with_index['LSOA']
        self.admissions = self.admissions_with_index['Admissions']

        # Load time/distance matrix
        self.travel_matrix = pd.read_csv('data/travel_matrix.csv', index_col=0)

        # Check data indices match
        self.check_loaded_data_indices_match()

        # Load initial population if data/load.csv exists
        try:
            self.loaded_population = np.loadtxt('data/load.csv', delimiter=',')
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), 'Loaded starting population from file')
        except:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), 'No initial population loaded from file')
        return


class Master():
    """
    Main algorithm code.
    1) Initialise algorithm object and load data
    """

    def __init__(self):
        """
        Set up algorithm environment.
        Load:
            Global variables
            Underlying data for algorithm:
                List of hospitals
                Patients by LSOA
                Travel matrix from all LSOA to all hospitals
        """
        # Set up class environment
        self.global_vars = GlobVariables()
        self.data = Data(self.global_vars.use_regions, self.global_vars.fix_hospital_status)
        self.pop = Pop()
        self.score = ScorePareto(self.global_vars.maximum_generations)
        self.generation = 0
        return

    def initialise_population(self):
        """
        This method creates a starting population.
        This may consist of a) a random population, b) a loaded population, or c) both.
        """
        self.pop.population = []

        if self.global_vars.initial_random_population_size > 0:
            self.pop.population = self.pop.create_random_population(
                self.global_vars.initial_random_population_size,
                self.data.hospitals,
                self.global_vars.fix_hospital_status,
                self.global_vars.use_regions,
                self.data.regions_dictionary)

            if len(self.data.loaded_population) > 0:
                self.pop.population = np.vstack((self.data.loaded_population, self.pop.population))
                self.pop.population = np.unique(self.pop.population, axis=0)
        return

    def run_algorithm(self):
        # Create initial population
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), 'Loading coffee and biscuits')
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), 'Starting generations')
        self.initialise_population()
        (self.pop.population, pareto_size) = self.select_population(0)  # 0 indicates base generation
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), 'Generation: %3.0f  Patients in target '
                                                                  'distance/admissions: %1.4f  Hamming: %1.4f  Pareto size: %6.0f' \
              % (0, self.score.progress_track[0], self.score.hamming[0], pareto_size))
        for gen in range(1, self.global_vars.maximum_generations):
            # Add new random population
            new_population_members_required = (int(self.pop.population.shape[
                                                       0] * self.global_vars.proportion_new_random_each_generation) + 1)
            new_population = self.pop.create_random_population(
                new_population_members_required,
                self.data.hospitals,
                self.global_vars.fix_hospital_status,
                self.global_vars.use_regions,
                self.data.regions_dictionary)

            # Combine populations before breeding
            self.pop.population = np.vstack((self.pop.population, new_population))

            # Get new children
            child_population = (self.pop.generate_child_population(
                self.global_vars.max_crossover_points,
                self.global_vars.mutation_rate,
                self.data.hospitals,
                self.global_vars.fix_hospital_status))

            # Combine populations
            self.pop.population = np.vstack((self.pop.population, child_population))

            # Remove scenarios with no hospitals
            check_hospitals = np.sum(self.pop.population, axis=1) > 0
            self.pop.population = self.pop.population[check_hospitals, :]

            # Remove non-unique rows
            self.pop.population = np.unique(self.pop.population, axis=0)
            

            # Select new Pareto front population
            (self.pop.population, pareto_size) = self.select_population(gen)

            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), 'Generation: %3.0f  Patients in target '
                                                                      'distance/admissions: %1.4f  Hamming: %1.4f  Pareto size: %6.0f' \
                  % (gen, self.score.progress_track[gen], self.score.hamming[gen], pareto_size))
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), 'End\n')
        return

    def save_pareto(self, scores, population, admissions):
        """Save latest Pareto front each generation"""
        scores_headers = (['#Hosp', 'Av_travel', 'Max_travel', 'Max_admissions', 'Min_admissions', 'Target_travel',
                           'Target_admissions', 'Target_travel+distance', '90th percentile travel',
                           '95th percentile travel', '99th percentile travel'])
        hospital_headers = self.data.hospitals.index
        scores_df = pd.DataFrame(scores, columns=scores_headers)
        scores_df.index.name = 'Index'
        population_df = pd.DataFrame(population, columns=hospital_headers)
        population_df.index.name = 'Index'
        admissions_df = pd.DataFrame(admissions, columns=hospital_headers)
        admissions_df.index.name = 'Index'
        scores_df.to_csv(self.global_vars.output_location + '/scores.csv')
        population_df.to_csv(self.global_vars.output_location + '/population.csv')
        admissions_df.to_csv(self.global_vars.output_location + '/admissions.csv')
        return

    def select_population(self, generation):
        # Score population
        (self.score.scores, self.score.hospital_admissions) = (
            self.score.score_pop(self.pop.population, self.data.travel_matrix.values,
                                 self.data.admissions,
                                 self.global_vars.target_travel,
                                 self.global_vars.target_min_admissions,
                                 self.data.hospital_count))

        # When regionalisation is selected remove solution where some patients are unallocated
        if self.global_vars.use_regions:
            mask = self.score.scores[:, 1] != np.inf
            if sum(mask) == 0:
                sys.exit("No feasible solutions in population. Exiting.")
            self.score.scores = self.score.scores[mask, :]
            self.score.hospital_admissions = self.score.hospital_admissions[mask, :]
            self.pop.population = self.pop.population[mask, :]

        # Check solution population is at least the minimum taregt population
        if self.score.scores.shape[0] < self.global_vars.minimum_population_size:
            # Population too small do not select from within population
            new_population = self.pop.population
            new_population_ids = np.arange(0, new_population.shape[0])
            new_population_scores = self.score.scores
            if len(new_population_ids) < 20000:
                hamming = np.average(pdist(new_population, 'hamming'))
            else:
                hamming = 999
            pareto_front = self.score.identify_pareto(new_population_scores, self.global_vars.pareto_scores_used)
            pareto_size = sum(pareto_front)
            self.save_pareto(new_population_scores, new_population, self.score.hospital_admissions)
            self.score.hamming[generation] = hamming
            print('No selection this generation; feasible solution population too small')

        else:
            unselected_scores = self.score.scores
            unselected_population_ids = np.arange(0, self.pop.population.shape[0])

            # First Pareto front
            pareto_front = self.score.identify_pareto(unselected_scores, self.global_vars.pareto_scores_used)
            selected_population_ids = unselected_population_ids[pareto_front]
            selected_scores = unselected_scores[pareto_front]

            # store first complete Pareto front
            pareto_population_ids = selected_population_ids
            pareto_size = len(selected_population_ids)
            pareto_population_scores = self.score.scores[selected_population_ids, :]
            pareto_population = self.pop.population[pareto_population_ids, :]
            pareto_hospital_admissions = self.score.hospital_admissions[pareto_population_ids, :]
            if len(selected_population_ids) < 20000:
                hamming = np.average(pdist(pareto_population, 'hamming'))
            else:
                hamming = 999
            self.score.hamming[generation] = hamming
            np.savetxt(self.global_vars.output_location + '/hamming.csv', self.score.hamming, delimiter=',')
            self.save_pareto(pareto_population_scores, pareto_population, pareto_hospital_admissions)

            # New population may need to be expanded or reduced depedning on required min/max size
            new_population_ids = pareto_population_ids  # This may be enlarged/reduced
            new_population_scores = selected_scores  # This may be enlarged/reduced
            unselected_population_ids = unselected_population_ids[np.invert(pareto_front)]
            unselected_scores = unselected_scores[np.invert(pareto_front)]

            # Check whether first Pareto front is within the population size required

            select_more = False

            if new_population_ids.shape[0] > self.global_vars.maximum_population_size:
                selection_size = self.global_vars.maximum_population_size
                (new_population_ids, new_population_scores) = self.score.reduce_by_crowding(
                    new_population_ids, new_population_scores, selection_size)

            if new_population_ids.shape[0] < self.global_vars.minimum_population_size:
                select_more = True
            while select_more:
                # Get next pareto front
                pareto_front = self.score.identify_pareto(unselected_scores, self.global_vars.pareto_scores_used)
                selected_population_ids = unselected_population_ids[pareto_front]
                selected_scores = unselected_scores[pareto_front]
                if new_population_ids.shape[0] + selected_population_ids.shape[0] > \
                        self.global_vars.minimum_population_size:
                    select_more = False
                    if new_population_ids.shape[0] + selected_population_ids.shape[0] < \
                            self.global_vars.maximum_population_size:
                        # New population smaller than maximum permitted
                        new_population_ids = np.hstack((new_population_ids, selected_population_ids))
                        new_population_scores = np.vstack((new_population_scores, selected_scores))
                    else:
                        # New population larger than permitted; reduce size
                        selection_size = self.global_vars.minimum_population_size - new_population_ids.shape[0]
                        (selected_population_ids, selected_scores) = self.score.reduce_by_crowding(
                            new_population_ids, new_population_scores, selection_size)
                        new_population_ids = np.hstack((new_population_ids, selected_population_ids))
                        new_population_scores = np.vstack((new_population_scores, selected_scores))
                else:
                    # Need to loop to select again; remove selected population/scores before re-looping
                    new_population_ids = np.hstack((new_population_ids, selected_population_ids))
                    new_population_scores = np.vstack((new_population_scores, selected_scores))
                    unselected_population_ids = unselected_population_ids[np.invert(pareto_front)]
                    unselected_scores = unselected_scores[np.invert(pareto_front)]

        # Print update and return
        new_population_ids = new_population_ids.astype(int)
        new_population = self.pop.population[new_population_ids, :]
        self.score.progress_track[generation] = max(
            new_population_scores[:, 7])  # record maximum in target range and admissions
        np.savetxt(self.global_vars.output_location + '/progress.csv', self.score.progress_track, delimiter=',')
        return (new_population, pareto_size)


class Pop():
    """
    Pop is the population class. It holds the current population and methods used for generating populations,
    including breeding.
    """

    def __init__(self):
        self.population = []  # holds the current population
        return

    def create_random_population(self, rows, hospitals, fix_hospitals, use_regions, regions_dictionary):
        """
        This method creates a starting population.
        This may consist of a) a random population, b) a loaded population, or c) both.
        For random populations, the method creates a population, removes duplicate rows, and repeats until
        the required number of non-duplicate rows are created.
        """
        rows_required = rows
        hospital_count = len(hospitals)
        population = []
        new_random_population = np.zeros((rows_required, hospital_count))  # create array of zeros

        if use_regions:
            while rows_required > 0:
                # When regions required populate so that all regions represented at least once
                for i in range(rows_required):
                    # Add in random number of hospitals
                    x = rn.randint(0, hospital_count)  # Number of 1s to add
                    new_random_population[i, 0:x] = 1  # Add requires 1s
                    np.random.shuffle(new_random_population[i])  # Shuffle the 1s randomly

                    # Pick one hospital from each region
                    for key, values in regions_dictionary.items():
                        pick_index = rn.randint(0, len(values) - 1)
                        new_random_population[i, values[pick_index]] = 1
                if fix_hospitals:
                    new_random_population = self.fix_hospital_status(hospitals, new_random_population)
                if len(population) == 0:  # first round of generating population
                    population = new_random_population
                else:
                    population = np.vstack((population, new_random_population))

                    # Remove any rows with no hospitals
                check_hospitals = np.sum(population, axis=1) > 0
                population = population[check_hospitals, :]

                # Remove non-unique rows
                population = np.unique(population, axis=0)
                population_size = population.shape[0]
                rows_required = 0 # rows - population_size
        else:
            # Method when regions not required
            while rows_required > 0:
                for i in range(rows_required):
                    x = rn.randint(1, hospital_count)  # Number of 1s to add
                    new_random_population[i, 0:x] = 1  # Add requires 1s
                    np.random.shuffle(new_random_population[i])  # Shuffle the 1s randomly
                if fix_hospitals:
                    new_random_population = self.fix_hospital_status(hospitals, new_random_population)
                if len(population) == 0:  # first round of generating population
                    population = new_random_population
                else:
                    population = np.vstack((population, new_random_population))

                # Remove any rows with no hospitals
                check_hospitals = np.sum(population, axis=1) > 0
                population = population[check_hospitals, :]

                # Remove non-unique rows
                population = np.unique(population, axis=0)
                population_size = population.shape[0]
                rows_required = 0
        return population

    @staticmethod
    def crossover(parents, max_crossover_points):

        chromsome_length = parents.shape[1]

        number_crossover_points = rn.randint(1, max_crossover_points)  # random, up to max
        # pick random crossover points in gene, avoid first position (zero position)
        crossover_points = rn.sample(range(1, chromsome_length), number_crossover_points)
        # appended zero at front for calucation of interval to first crossover
        crossover_points = np.append([0], np.sort(crossover_points))
        # create intervals of ones and zeros
        intervals = crossover_points[1:] - crossover_points[:-1]
        # Add last interval
        intervals = np.append([intervals], [chromsome_length - np.amax(crossover_points)])
        # Build boolean arrays for cross-overs
        current_bool = True  # sub sections will be made up of repeats of boolean true or false, start with true
        # empty list required for append
        selection1 = []

        for interval in intervals:  # interval is the interval between crossoevrs (stored in 'intervals')
            new_section = np.repeat(current_bool, interval)  # create subsection of true or false
            current_bool = not current_bool  # swap true to false and vice versa
            selection1 = np.append([selection1], [new_section])  # add the new section to the existing array

        selection1 = np.array([selection1], dtype=bool)
        selection2 = np.invert(selection1)  # invert boolean selection for second cross-over product

        child_1 = np.choose(selection1, parents)  # choose from parents based on selection vector
        child_2 = np.choose(selection2, parents)

        children = np.append(child_1, child_2, axis=0)

        return children

    @staticmethod
    def fix_hospital_status(hospitals, population):
        """
        Fixes hospitals to be forced open or forced closed as required.
        This is done by overlaying a matrix of forced open (1) or forced closed (-1)
        """
        fix_list = hospitals['Fixed'].values
        population_size = population.shape[0]
        fix_matrix = np.array([fix_list, ] * population_size)
        population[fix_matrix == 1] = 1  # Fixes the open hospitals to have a value 1
        population[fix_matrix == -1] = 0  # Fixes the closed hospitals to have a value 0
        return population

    def generate_child_population(self, maximum_crossovers, mutation, hospitals, fix_hospitals):
        pop_required = self.population.shape[0]  # 2 children per breeding round
        hospital_count = self.population.shape[1]
        child_population = np.zeros((0, self.population.shape[1]))
        population_size = self.population.shape[0]
        for i in range(pop_required):
            parent1_ID = rn.randint(0, population_size - 1)  # select parent ID at random
            parent2_ID = rn.randint(0, population_size - 1)  # select parent ID at random
            parents = np.vstack((self.population[parent1_ID], self.population[parent2_ID]))
            children = self.crossover(parents, maximum_crossovers)
            child_population = np.vstack((child_population, children))
        # Apply random mutation
        random_mutation_array = np.random.random(
            size=(child_population.shape))
        
        random_mutation_boolean = \
            random_mutation_array <= mutation

        child_population[random_mutation_boolean] = \
            np.logical_not(child_population[random_mutation_boolean])
	
	    # Fix hospital status if required
        if fix_hospitals:
            child_population = self.fix_hospital_status(hospitals, child_population)
        # Remove any rows with no hospitals
        check_hospitals = np.sum(child_population, axis=1) > 0
        child_population = child_population[check_hospitals, :]
        return child_population


class ScorePareto():
    """
    This class holds the methods for scoring solutions and identifying pareto fronts

    # Score_matrix:
    # 0: Number of hospitals
    # 1: Average distance
    # 2: Maximum distance
    # 3: Maximum admissions to any one hospital
    # 4: Minimum admissions to any one hospital
    # 5: Proportion patients within target time/distance
    # 6: Proportion patients attending unit with target admission numbers
    # 7: Proportion of patients meeting distance and admissions target
    # 8: 90th percentile travel
    # 9: 95th percentile travel
    # 10: 99th percentile travel

    """

    def __init__(self, generations):
        self.scores = []  # holds the scores for the current population
        self.pareto_front = []
        self.hospital_admissions = []
        self.progress_track = np.zeros((generations))
        self.hamming = np.zeros((generations))
        return

    @staticmethod
    def calculate_crowding(scores):
        # Crowding is based on chrmosome scores (not chromosome binary values)
        # All scores are normalised between low and high
        # For any one score, all solutions are sorted in order low to high
        # Crowding for chromsome x for that score is the difference between th enext highest and next lowest score
        # Total crowding value sums all crowding for all scores
        population_size = len(scores[:, 0])
        number_of_scores = len(scores[0, :])
        # create crowding matrix of population (row) and score (column)
        crowding_matrix = np.zeros((population_size, number_of_scores))
        # normalise scores
        normed_scores = (scores - scores.min(0)) / scores.ptp(0)  # numpy ptp is range (max-min)
        # Calculate crowding
        for col in range(number_of_scores):  # calculate crowding distance for each score in turn
            crowding = np.zeros(population_size)  # One dimensional array
            crowding[0] = 1  # end points have maximum crowding
            crowding[population_size - 1] = 1  # end points have maximum crowding
            sorted_scores = np.sort(normed_scores[:, col])  # sort scores
            sorted_scores_index = np.argsort(normed_scores[:, col])  # index of sorted scores
            # Calculate crowding distance for each individual
            crowding[1:population_size - 1] = sorted_scores[2:population_size] - sorted_scores[
                                                                                 0:population_size - 2]
            re_sort_order = np.argsort(sorted_scores_index)  # re-sort to original order step 1
            sorted_crowding = crowding[re_sort_order]  # re-sort to orginal order step 2
            crowding_matrix[:, col] = sorted_crowding  # record crowding distances
        crowding_distances = np.sum(crowding_matrix, axis=1)  # Sum croding distances of all scores
        return crowding_distances

    @staticmethod
    def identify_pareto(scores, used_scores):
        scores_for_pareto = np.copy(scores)
        # Pareto front assumed higher is better, therefore inverse scores where lower is better
        scores_for_pareto[:, 0] = 1 / scores_for_pareto[:, 0]  # Number of hospitals
        scores_for_pareto[:, 1] = 1 / scores_for_pareto[:, 1]  # Average distance
        scores_for_pareto[:, 2] = 1 / scores_for_pareto[:, 2]  # Maximum distance
        scores_for_pareto[:, 3] = 1 / scores_for_pareto[:, 3]  # Maximum admissions
        scores_for_pareto[:, 8] = 1 / scores_for_pareto[:, 8]  # 90th percentile
        scores_for_pareto[:, 9] = 1 / scores_for_pareto[:, 9]  # 95th percentile
        scores_for_pareto[:, 10] = 1 / scores_for_pareto[:, 10]  # 99th percentile

        scores_for_pareto = scores_for_pareto[:, used_scores]

        population_size = scores_for_pareto.shape[0]
        pareto_front = np.ones(population_size, dtype=bool)
        for i in range(population_size):
            for j in range(population_size):
                if all(scores_for_pareto[j] >= scores_for_pareto[i]) and any(
                        scores_for_pareto[j] > scores_for_pareto[i]):
                    # j dominates i
                    pareto_front[i] = 0
                    break
        return pareto_front

    def reduce_by_crowding(self, population_ids, scores, number_to_select):
        # This function selects a number of solutions based on tournament of crowding distances
        # Two members of the population ar epicked at random
        # The one with the higher croding dostance is always picked
        crowding_distances = self.calculate_crowding(scores)  # crowding distances for each member of the population
        picked_population_ids = np.zeros((number_to_select))  # array of picked solutions (actual solution not ID)
        picked_scores = np.zeros((number_to_select, len(scores[0, :])))  # array of scores for picked solutions
        for i in range(number_to_select):
            population_size = population_ids.shape[0]
            fighter1ID = rn.randint(0, population_size - 1)  # 1st random ID
            fighter2ID = rn.randint(0, population_size - 1)  # 2nd random ID
            if crowding_distances[fighter1ID] >= crowding_distances[fighter2ID]:  # 1st solution picked
                picked_population_ids[i] = population_ids[fighter1ID]  # add solution to picked solutions array
                picked_scores[i, :] = scores[fighter1ID, :]  # add score to picked solutions array
                # remove selected solution from available solutions
                population_ids = np.delete(population_ids, (fighter1ID), axis=0)  # remove picked solution
                scores = np.delete(scores, (fighter1ID), axis=0)  # remove picked score (as line above)
                crowding_distances = np.delete(crowding_distances, (fighter1ID), axis=0)  # remove crowding score
            else:  # solution 2 is better. Code as above for 1st solution winning
                picked_population_ids[i] = population_ids[fighter2ID]
                picked_scores[i, :] = scores[fighter2ID, :]
                population_ids = np.delete(population_ids, (fighter2ID), axis=0)
                scores = np.delete(scores, (fighter2ID), axis=0)
                crowding_distances = np.delete(crowding_distances, (fighter2ID), axis=0)
        return (picked_population_ids, picked_scores)

    def score_pop(self, population, matrix, node_admissions, target_travel, target_admissions, hospital_count):

        # Create empty score matrix
        number_score_parameters = 10
        population_size = population.shape[0]
        hospital_admissions_matrix = np.zeros((population_size, hospital_count))

        score_matrix = np.zeros((population_size, number_score_parameters + 1))

        for scenario in range(population_size):  # Loop through population of solutions
            node_results = np.zeros((len(node_admissions), 6))
            # Node results stores results by patient node. These are used in the calculation of results
            # Col 0: Distance to closest hospital
            # Col 1: Node within target travel (boolean)
            # Col 2: Hospital ID
            # Col 3: Number of admissions to hospital ID
            # Col 4: Does hospital meet admissions target (boolean)
            # Col 5: Admissions and travel target (boolean)

            # Calculate average distance
            mask = np.array(population[scenario], dtype=bool)
            masked_matrix = matrix[:, mask]
            node_results[:, 0] = np.amin(masked_matrix, axis=1)  # distance to closest hospital
            node_results[:, 1] = node_results[:, 0] <= target_travel  # =1 if target distance 1 met

            # Identify closest hospital
            closest_hospital_ID = np.argmin(masked_matrix, axis=1)  # index of closest hospital
            node_results[:, 2] = closest_hospital_ID

            # Create matrix of number of admissions to each hospital (also returned from method)
            admitting_hospitals = np.where(population[scenario] == 1)
            hospital_admissions = np.bincount(closest_hospital_ID,
                                              weights=node_admissions,
                                              minlength=len(admitting_hospitals[0]))

            hospital_admissions_matrix[scenario, admitting_hospitals] = hospital_admissions

            # Lookup admissions to hospital
            node_results[:, 3] = np.take(hospital_admissions, closest_hospital_ID)
            node_results[:, 4] = node_results[:, 3] >= target_admissions

            # Check target distance and hospital admissions
            node_results[:, 5] = (node_results[:, 1] == 1) & (node_results[:, 4] == 1)

            # Calculate summed results from node results
            # ------------------------------------------

            # Calculate number of hospitals
            score_matrix[scenario, 0] = sum(population[scenario])

            # Calculate average distance by multiplying node distance * admission numbers and divide by total admissions
            total_admissions = np.sum(node_admissions)
            weighted_distances = np.multiply(node_results[:, 0], node_admissions)
            average_distance = np.sum(weighted_distances) / total_admissions
            score_matrix[scenario, 1] = average_distance
            score_matrix[scenario, 2] = np.max(node_results[:, 0])

            # Max, min and max/min number of admissions to each hospital
            score_matrix[scenario, 3] = np.max(hospital_admissions)
            score_matrix[scenario, 4] = np.min(hospital_admissions)

            # Calculate proportion patients within target travel and admissions
            score_matrix[scenario, 5] = np.sum(node_admissions[node_results[:, 1] == 1]) / total_admissions
            score_matrix[scenario, 6] = np.sum(node_admissions[node_results[:, 4] == 1]) / total_admissions
            score_matrix[scenario, 7] = np.sum(node_admissions[node_results[:, 5] == 1]) / total_admissions

            # Get weighted percentiles
            wp = self.weighted_percentiles(node_results[:, 0], node_admissions, [0.90, 0.95, 0.99])
            score_matrix[scenario, 8] = wp[0]
            score_matrix[scenario, 9] = wp[1]
            score_matrix[scenario, 10] = wp[2]

        return (score_matrix, hospital_admissions_matrix)

    @staticmethod
    def weighted_percentiles(data, wt, percentiles):
        assert np.greater_equal(percentiles, 0.0).all(), "Percentiles less than zero"
        assert np.less_equal(percentiles, 1.0).all(), "Percentiles greater than one"
        data = np.asarray(data)
        assert len(data.shape) == 1
        if wt is None:
            wt = np.ones(data.shape, np.float)
        else:
            wt = np.asarray(wt, np.float)
            assert wt.shape == data.shape
            assert np.greater_equal(wt, 0.0).all(), "Not all weights are non-negative."
        assert len(wt.shape) == 1
        n = data.shape[0]
        assert n > 0
        i = np.argsort(data)
        sd = np.take(data, i, axis=0)
        sw = np.take(wt, i, axis=0)
        aw = np.add.accumulate(sw)
        if not aw[-1] > 0:
            raise ValueError('Nonpositive weight sum')
        w = (aw - 0.5 * sw) / aw[-1]
        spots = np.searchsorted(w, percentiles)
        o = []
        for (s, p) in zip(spots, percentiles):
            if s == 0:
                o.append(sd[0])
            elif s == n:
                o.append(sd[n - 1])
            else:
                f1 = (w[s] - p) / (w[s] - w[s - 1])
                f2 = (p - w[s - 1]) / (w[s] - w[s - 1])
                assert f1 >= 0 and f2 >= 0 and f1 <= 1 and f2 <= 1
                assert abs(f1 + f2 - 1.0) < 1e-6
                o.append(sd[s - 1] * f1 + sd[s] * f2)
        return o

# CODE ENTRY POINT #
# ===================

if __name__ == '__main__':
    print()
    model = Master()
    model.run_algorithm()
