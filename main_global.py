# from generate_mtrx_global.py import generate_matrix
# from simulate_global.py import run_simulation


if __name__== "__main__":

    dataset = 'msd' # dataset tag

    # Execute generate matrix_global
    generate_matrix(dataset)

    for i in ['als', 'bpr', 'lmf']:
        for j in ['topm', 'rand']:

            # Execute simulate_global
            run_simulation(dataset, i, j)
            # enviar al open la info de i y j, with open(f'output_{i}_{j}_{split_folder}.txt', 'w') as output_file: