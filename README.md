# pomalu, ale jistĕ ...
... I want to play with PyTorch and Pandas to chew on some timeseries like stock market data.

Getting real stock market data can be a challenge.
I will use the Tiingo data service to retrieve the end of day (EOD) historical data for training and testing purposes.
Tiingo offers a free subscription limited by e.g. volume and count of requests.
It still supports REST and Websocket APIs to get the data as we need.
Long story short, Tiingo seems to be simply: Geschleckte gut genug!
Further details can be found here: https://api.tiingo.com

# My way to go ...
First I want to create a simple auto encoder to memorize possible patterns in stock market data.
Training this net should be solveable by using unsupervised learning.
Second I'm going to train a net to classify compare results from input and output of the auto encoder using supervised learning.
Finally I will try to train a "trading" agent (constructed from auto encoder, classifier and an additional net self). I give reinforcement learning a try.
My personal target is to build a net, where I can throw some historical end of day data and watch how it would trade on other unknown test data beside the training data.
Maybe the results are going to be BAD, maybe the results are going to be good. I don't know!

# WARNING:
Some events and data could be already priced into the historical data and may build a possible indicator to predict something in the future.
The TRUE story is: The STOCK MARKET is a very COMPLEX and VOLATILE BEAST! There exists a myriad of factors, like events, data, crisises and even the actual place of the mond at this moment will influence the market, which may result in a HIGH LOSS or minimal gain of a particular stock.
One thing is save, the project will NEVER BE the magic money maker to beat the stock market and it is NOT A WARRANTY FOR THE INVESTMENT DECISIONS YOU WILL MAKE!
YOU ARE SELF RESPONSIBLE FOR YOUR INVESTMENTS BASED ON YOUR OWN DECISIONS, NOT I NOR THE SOFTWARE!

The project and the resulting agent is just a simple TRIAL and ERROR, "I'll give it a try!" and will DEFINITELY NOT BEAT THE MARKET. It is just to play with some time series data...

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
