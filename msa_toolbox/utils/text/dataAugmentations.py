import nlpaug.augmenter.word as naw
import nlpaug.flow as naf

def Back_translation(text):
    aug = naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-de',
        to_model_name='Helsinki-NLP/opus-mt-de-en',
        device='cuda'
    )
    augmented_text = aug.augment(text)
    return augmented_text

def Random_Swap(text , p = 0.1):
    aug = naw.RandomWordAug(action="swap" , aug_p = p)
    augmented_text = aug.augment(text)
    return augmented_text

def Random_Insert(text , p = 0.1):
    aug = naw.RandomWordAug(action="insert" , aug_p = p)
    augmented_text = aug.augment(text)
    return augmented_text

def Random_Delete(text , p = 0.1):
    aug = naw.RandomWordAug(action="delete" , aug_p = p)
    augmented_text = aug.augment(text)
    return augmented_text

def Random_Substitute(text , p = 0.1):
    aug = naw.RandomWordAug(action="substitute" , aug_p = p)
    augmented_text = aug.augment(text)
    return augmented_text

def Random_Synonym(text , p = 0.1):
    aug = naw.SynonymAug(aug_src='wordnet' , aug_p = p)
    augmented_text = aug.augment(text)
    return augmented_text