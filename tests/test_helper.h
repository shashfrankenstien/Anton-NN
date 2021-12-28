
#include "nn.h"

template <class N, typename I = double, typename L>
void run_test(Net<N> &myNet, std::vector<std::vector<I>> &test_inputs, std::vector<std::vector<double>> &test_outputs, unsigned repetitions, L print_inputs)//void (*print_inputs)(unsigned idx))
{
    std::vector<double> results_container;
    double abs_avg_error;

    unsigned counter = 0;
    for (unsigned r = 1; r <= repetitions; r++) {
        for (unsigned i = 0; i < test_inputs.size(); i++) {
            myNet.feed_forward(test_inputs[i]);
            myNet.back_propagate_sgd(test_outputs[i]);

            printf("[%d] inp: ", counter++);

            print_inputs(i);
            // if (opt==MNIST) {
            //     printf("%d ", uint8_reconstruct(test_outputs[i]));
            // }

            printf(" exp: ");
            for (unsigned j = 0; j < test_outputs[i].size(); j++)
                printf("%d ", (int)test_outputs[i][j]);

            printf(" -> ");
            results_container.clear();
            abs_avg_error = 0;
            myNet.get_results(results_container, abs_avg_error);

            bool overall_success = true;
            printf("RES: ");
            for (unsigned n = 0; n < results_container.size(); n++) {
                int bin_out = (results_container[n]>=0.5) ? 1 : 0;
                printf("%f(%d) ", results_container[n], bin_out);
                printf("%c ", (bin_out!=test_outputs[i][n]) ? 'x' : ' ');
                overall_success = overall_success && (bin_out==test_outputs[i][n]);
            }
            printf("ERR: %3.4f %c\n", abs_avg_error*100, (overall_success? '-': 'x'));
            // printf("**************\n");
        }
    }
}
