from rag import FaissStore, FeedbackRAG, Settings



sett = Settings()
vec = FaissStore(settings=sett)
rag = FeedbackRAG(sett, vec)




#############create csv like the test format to enter into the vec db


if __name__ == "__main__":

    dataset_path = "dataset\untouched.csv"
    rag.test(dataset_path)