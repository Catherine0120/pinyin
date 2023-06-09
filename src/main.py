import argparse
from graph_generator_bi import bi_gram_generator
from graph_generator_bi_plus import bi_gram_generator_plus
from graph_generator_bi2 import bi_gram_generator_2
from graph_generator_tri import tri_gram_generator
from graph_generator_tri2 import tri_gram_generator_2
from validator import test

def run(model: str, input_path: str, output_path: str, t: bool, std_output: str):
    """ main entrance on the program """
    if t and std_output == "none":
        print("[Error]: Please provides std_output file path")
        return
    if model == "bigram_2":
        bi_gram_generator_2(input_path, output_path)
    if model == "bigram":
        bi_gram_generator(input_path, output_path)
    if model == "bigram+":
        bi_gram_generator_plus(input_path, output_path)
    if model == "trigram":
        tri_gram_generator(input_path, output_path)
    if model == "trigram_2":
        tri_gram_generator_2(input_path, output_path)
    if t:
        test(output_path, std_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="trigram", 
                        choices=["bigram", "bigram_2", "bigram+","trigram", "trigram_2"], 
                        help="choose one model to run")
    parser.add_argument("-i", "--input", type=str, default="input.txt", help="choose an input file")
    parser.add_argument("-o", "--output", type=str, default="output.txt", help="choose an output file")
    parser.add_argument("-t", "--test", action="store_true", help="choose whether to test the result")
    parser.add_argument("-f", "--std_output", type=str, default="none", help="give a std_output for testing")
    args = parser.parse_args()
    run(args.model, args.input, args.output, args.test, args.std_output)