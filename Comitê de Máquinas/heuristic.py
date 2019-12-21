# Autor: Pedro Antonio Fernades da Silva

'''Método heurístico para seleção dos melhores atributos'''

from random import random
from random import randint
import numpy as np
import tensorflow as tf


class GeneticAlgorithm:
    '''Aplicação de algoritmo genético para seleção dos melhores atributos

    Parâmetros
    ----------
    model: Rede neural usada na heurística

    n_generations: Número total de gerações

    population_size: Tamanho da população em cada geração

    chromosome_size: Tamanho do cromossomo

    mutation_rate: Taxa de mutação em cada gene

    crossover_rate: Taxa de sucesso para um cruzamento

    Atributos
    ---------
    evaluations: Avaliação de cada indivíduo da população

    evaluation_sum: Somatório de todas as avaliações da geração

    current_generation: Geração atual

    current_models_generation: Rede neural de cada indivíduo da geração atual

    best_model: Melhor rede neural da geração

    best_evaluation: Melhor avaliação da geração

    best_chromossome: Melhor cromossomo

    '''

    def __init__(self, model, n_generations, population_size,
                 chromosome_size, mutation_rate, crossover_rate):

        self.n_generations = n_generations
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.evaluations = []
        self.evaluation_sum = 0

        self.current_generation = []
        self.current_models_generation = []
        # model.save_weights('save_weights.h5')

        self.best_model = None
        self.best_evaluation = -1
        self.best_chromossome = None

        for i in range(population_size):
            self.current_generation.append(np.random.choice(
                a=[False, True], size=chromosome_size))

            if sum(self.current_generation[i]) == 0:
                del self.current_generation[i]

            self.current_models_generation.append(
                tf.keras.models.clone_model(model))

    def fit(self, x, y):
        '''Inicia a heurística

        Parâmetros
        ----------

        x: Atributos da base de dados

        y: Atributo classe da base de dados

        '''
        for i in range(self.n_generations):
            self.best_model = None
            self.best_evaluation = -1
            self.best_chromossome = None
            print("========================== " + str(i) +
                  " gen ==========================")
            self.evaluate(x, y)
            self.roulette()
        return

    def crossover(self, next_generation_parents):
        '''Operação de cruzamento entre o selecionados na roleta

        Parâmetros
        ----------

        next_generation_parents: Os geradores da próxima geração, ou seja, os pais

        '''
        next_generation = []
        next_models_generation = []

        for parents, model in zip(next_generation_parents, self.current_models_generation):

            parent1_chromosome = self.current_generation[parents[0]]
            parent2_chromosome = self.current_generation[parents[1]]

            half_chromosome1 = parent1_chromosome[:int(self.chromosome_size/2)]
            half_chromosome2 = parent2_chromosome[int(self.chromosome_size /
                                                      2): self.chromosome_size]

            sons_chromosome = np.concatenate(
                [half_chromosome1, half_chromosome2])

            self.mutation(sons_chromosome)

            if sum(sons_chromosome) == 0:
                sons_chromosome[randint(0, sons_chromosome.shape)] = True

            next_generation.append(sons_chromosome)
            next_models_generation.append(tf.keras.models.clone_model(model))

        self.current_generation = next_generation
        self.current_models_generation = next_models_generation
        self.evaluation_sum = 0
        self.evaluations = []

    def mutation(self, chromosome):
        '''Operação de mutação nos genes de um cromossomo

        Parâmetros
        ----------

        chromosome: Um cromossomo que podera ou não ter mutações

        '''
        for gene in chromosome:
            if random() < self.mutation_rate:
                gene = not gene

    def roulette(self):
        '''Operação de roleta para seleção dos indivídous que terão descendência'''
        roullete_threshold = []

        previous = 0
        for evaluation in self.evaluations:
            previous += evaluation/self.evaluation_sum
            roullete_threshold.append(previous)

        next_generation_parents = []

        while len(next_generation_parents) < self.population_size:
            arrow1 = random()
            arrow2 = random()
            parent1 = -1
            parent2 = -1

            for i in range(len(roullete_threshold)):
                if(arrow1 < roullete_threshold[i] and parent1 == -1):
                    parent1 = i
                if(arrow2 < roullete_threshold[i] and parent2 == -1):
                    parent2 = i
                if(parent1 != -1 and parent2 != -1):
                    break

            if random() < self.crossover_rate:
                next_generation_parents.append((parent1, parent2))

        self.crossover(next_generation_parents)

    def evaluate(self, x, y):
        '''Avaliação de cada modelo da geração atual

        Parâmetros
        ----------

        x: Atributos da base de dados

        y: Atributos classe da base de dados

        '''
        for chromosome, model in zip(self.current_generation, self.current_models_generation):
            x_by_chromosome = x[x.columns[chromosome]]

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            history = model.fit(x_by_chromosome.to_numpy(
            ), y.to_numpy(), verbose=0, batch_size=10, epochs=20)

            accuracy_history = history.history['accuracy']
            accuracy = accuracy_history[len(accuracy_history)-1]

            self.evaluations.append(accuracy)
            self.evaluation_sum += accuracy

            if self.best_evaluation < accuracy:
                self.best_evaluation = accuracy
                self.best_model = model
                self.best_chromossome = chromosome

        print('- Best Chromossome' + str(self.best_chromossome) +
              ' \n       Acurracy: ' + str(self.best_evaluation))
