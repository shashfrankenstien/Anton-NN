#include <deque>
#include "conf.h"
#include "nn.h"
#include "test_helper.h"


void recurrent_bit_series(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs, unsigned n_samples)
{
    std::deque<double> d = {1,0,1,0,0,0,1,0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,0,1};
    for (int i = 0; i < n_samples; i++) {

        inputs.push_back({d[0], d[1], d[2], d[3]});
        outputs.push_back({d[4], d[5], d[6], d[7]});

        // rotate
        double val = d.back();
        d.pop_back();
        d.push_front(val);
    }
}



void one_hot_word_stream(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs, unsigned n_samples)
{
    // jack = 100000
    // tom = 010000
    // jerry = 001000
    // cheese = 000100
    // saw = 000010
    // FULLSTOP = 000001

    std::vector<std::vector<double>> choices = {
        // jack saw tom.
        {1,0,0,0,0,0}, {0,0,0,0,1,0}, {0,1,0,0,0,0}, {0,0,0,0,0,1},
        // tom saw jerry.
        {0,1,0,0,0,0}, {0,0,0,0,1,0}, {0,0,1,0,0,0}, {0,0,0,0,0,1},
        // jerry saw cheese.
        {0,0,1,0,0,0}, {0,0,0,0,1,0}, {0,0,0,1,0,0}, {0,0,0,0,0,1},
        // tom saw jack.
        {0,1,0,0,0,0}, {0,0,0,0,1,0}, {1,0,0,0,0,0}, {0,0,0,0,0,1},
        // jerry saw jack.
        {0,0,1,0,0,0}, {0,0,0,0,1,0}, {1,0,0,0,0,0}, {0,0,0,0,0,1},
    };

    int siz = choices.size();
    for (int i = 0; i < n_samples; i++) {

        int inp = i % siz;
        int oup = (i+1) % siz;
        inputs.push_back(choices[inp]);
        outputs.push_back(choices[oup]);
    }
}


#include <fstream> // std::ifstream
#include <sstream> // std::stringstream

#include <vector>
#include <map>
#include <set>

typedef std::map<std::string, std::vector<double>> encoding_t;


class TextTrainer {

    public:
        TextTrainer(const char* filepath);
        void get_training_set(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs, unsigned n_samples);

        std::vector<double> encode(const std::string& raw);
        std::string decode(const std::vector<double>& encoded);

    private:
        std::vector<std::string> tokens;
        std::set<std::string> encoding;
        encoding_t lookup;

        void tokenize(const char* filepath);

        static unsigned nbits_to_represent(unsigned num);
        static void int_bittify(unsigned val, unsigned nbits, std::vector<double> &res);
        static unsigned int_reconstruct(const std::vector<double> &res);

};

// number of bits required to represent num
unsigned TextTrainer::nbits_to_represent(unsigned num)
{
    return (log(num) / log(2)) + 1;
}

// convert int to an array of 1s and 0s that represent the binary value of the int
void TextTrainer::int_bittify(unsigned val, unsigned nbits, std::vector<double> &res)
{
    for (unsigned i = 0; i < nbits; i++) {
        res.push_back((val & 0x01 == 0x01) ? 1 : 0);
        val >>= 1;
    }
}
// convert an array of 1s and 0s back to an int
unsigned TextTrainer::int_reconstruct(const  std::vector<double> &res)
{
    unsigned val = 0x00;
    for (int i = res.size()-1; i >= 0; i--) {
        // res.push_back((val & 0x01 == 0x01) ? 1 : 0);
        val |= (((res[i]>0) ? 0x01 : 0x00));
        if(i>0)
            val <<= 1;
    }
    return val;
}


TextTrainer::TextTrainer(const char* filepath)
{
    tokenize(filepath);
    int idx = 0;
    unsigned total_words = encoding.size();
    unsigned nbits = nbits_to_represent(total_words);
    for (std::set<std::string>::iterator it = encoding.begin(); it != encoding.end(); ++it) {
        std::vector<double> val;
        int_bittify(idx, nbits, val);
        lookup[*it] = val;
        printf("%d - ", idx++);
        for (unsigned v = 0; v < val.size(); v++)
            printf("%.0f", val[v]);
        printf(" - %s\n", (*it).c_str());
    }
}

void TextTrainer::tokenize(const char* filepath)
{
    std::ifstream f;
    std::stringstream stream;
    std::string txt;
    f.open(filepath);

    if (f.is_open()) {
        stream << f.rdbuf();
        txt = stream.str();
        // printf("%s\n", txt.c_str());
    } else {
        printf("open failed");
        return;
    }

    f.close();

    std::string tmp = "";
    for (unsigned pos = 0; pos < txt.size(); pos++) {
        if (txt[pos]=='.' || txt[pos]==',' || txt[pos]==' ' || txt[pos]=='?' || txt[pos]=='"' || txt[pos]=='\n') {
            if (tmp.size() > 0) {
                tokens.push_back(tmp);
                encoding.insert(tmp);
            }
            tmp.clear();
            if (txt[pos]=='.') {
                tokens.push_back(".");
                encoding.insert(".");
            } else if (txt[pos]==',') {
                tokens.push_back(",");
                encoding.insert(",");
            } else if (txt[pos]=='?') {
                tokens.push_back("?");
                encoding.insert("?");
            } else if (txt[pos]=='"') {
                tokens.push_back("\"");
                encoding.insert("\"");
            }
        } else
            tmp.push_back(std::tolower(txt[pos]));
    }

    if (tmp.size() > 0) {
        tokens.push_back(tmp);
        encoding.insert(tmp);
    }
}




void TextTrainer::get_training_set(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs, unsigned n_samples)
{

    int siz = tokens.size();
    for (int i = 0; i < n_samples; i++) {

        int inp = i % siz;
        int oup = (i+1) % siz;
        inputs.push_back(lookup[tokens[inp]]);
        outputs.push_back(lookup[tokens[oup]]);

        // int inp = i % siz;
        // int inp2 = (i+1) % siz;
        // int oup = (i+2) % siz;
        // std::vector<double> input = lookup[tokens[inp]];
        // input.insert( input.end(), lookup[tokens[inp2]].begin(), lookup[tokens[inp2]].end() );
        // inputs.push_back(input);
        // outputs.push_back(lookup[tokens[oup]]);
    }
}


std::vector<double> TextTrainer::encode(const std::string& raw)
{
    return lookup[raw];
}
std::string TextTrainer::decode(const std::vector<double>& encoded)
{
    unsigned idx = TextTrainer::int_reconstruct(encoded);
    if (idx >= encoding.size())
        return "_";
    return *std::next(encoding.begin(), idx);
}





void input_printer(std::vector<double> &arr)
{
    for (unsigned j = 0; j < arr.size(); j++)
        printf("%d", (int)arr[j]);
}


int main()
{
    unsigned samp_size = 700000;
    unsigned repetitions = 1;

    srand(RANDOM_SEED);

    std::vector<std::vector<double>> test_inputs;
    std::vector<std::vector<double>> test_outputs;

    // const char* datafile = "data/text_datasets/blind_men_and_elephant.txt";
    // const char* datafile = "data/text_datasets/sticks.txt";
    // const char* datafile = "data/text_datasets/alphabet.txt";
    // const char* datafile = "data/text_datasets/buzzy_the_bee.txt";
    const char* datafile = "data/text_datasets/trumptweets/trumptweets.txt";
    TextTrainer txt(datafile);
    txt.get_training_set(test_inputs, test_outputs, samp_size);

    // std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 160, 80, 40, (unsigned)test_outputs[0].size()}; // blind_men_and_elephant 95.9% with small random weights and 500k training set
    // std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 160, 80, 40, (unsigned)test_outputs[0].size()}; // sticks 97.2% with small random weights
    std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 160, 80, 40, (unsigned)test_outputs[0].size()}; // trump tweets 93.2% with small random weights and 700k samples
    // std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 20, 20, (unsigned)test_outputs[0].size()}; // 86.25% on alphabet with small random weights
    // std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 20, 20, (unsigned)test_outputs[0].size()}; // 96.15% on buzzy with small random weights
    // std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 100, 50, 50, 20, 10, (unsigned)test_outputs[0].size()}; // 98% on buzzy with large random weights!
    Net<RecurrentNeuron> anton_nn(layers);
    train_net<RecurrentNeuron>(anton_nn, test_inputs, test_outputs, repetitions, [&test_inputs](unsigned idx){input_printer(test_inputs[idx]);});


    std::vector<double> word = test_inputs[0];
    double err;

    for (unsigned k=0;k<word.size();k++)
        printf("%f", word[k]);
    printf("\n");

    for (unsigned i = 0; i < 80; i++) {
        std::string word_str = txt.decode(word);
        printf("%s ", word_str.c_str());
        anton_nn.feed_forward(word);
        word.clear();
        anton_nn.get_results(word, err, 0.5);
    }

    printf("\n");

}
