import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Add your description here.
 */
class SimilarSounds
{
	// ******DO NO CHANGE********//
		
	/**
	 * wordToSound maps each word to its corresponding sound.
     */
	static Map<String, String> wordToSound;
	
	/**
	 * soundGroupToSimilarWords maps each sound-group to a BST containing all the words that share that sound-group.
     */
	static Map<String, BST<String>> soundGroupToSimilarWords;
	
	//just added
	// static Map<String, BST<String>> soundgroupSimilarWordsInList;
	// static Map<String, Integer> checker;
	/**
	 * Do not change.
	 * @param words one or more words passed on the command line.
     */		
	public static void processWords(String words[]) {
			
		ArrayList<String> lines = (ArrayList<String>)Extractor.readFile("word_to_sound.txt");
		populateWordToSoundMap(lines);
		populateSoundGroupToSimilarWordsMap(lines);
		
		if (words.length >= 2) {
			// check which of the words in the list have matching sounds 
			findSimilarWordsInList(words); 
		} else if (words.length == 1) {
			// get the list of words with matching sounds as this word
			findSimilarWordsTo(words[0]);
		} 
	}
	
	/**
	 *  Main Method.
	 *  
	 *  @param args args
	 */
	public static void main(String args[]) {
		if (args.length == 0) {
			System.out.println("Wrong number of arguments, expecting:");
			System.out.println("java SimilarSounds word1 word2 word3...");
			System.out.println("java SimilarSounds word");
			System.exit(-1);
		} 
		
		wordToSound = new java.util.HashMap<>(); // maps <word, sound>
        soundGroupToSimilarWords = new java.util.HashMap<>(); // maps <sound-group, sorted list of words with similar sounds>
		
		processWords(args);

	}
	// ******DO NO CHANGE********//
	
	
	
	
	/**
	 * Given a list of all entries in the database, this method populates the wordToSound map
	 * as follows: the key is the word, and the value is the sound (i.e., the sequence of unisounds)
	 * For example, if the line entry is "moderated M AA1 D ER0 EY2 T IH0 D", the key would be "moderated"
	 * and the value would be "M AA1 D ER0 EY2 T IH0 D"
	 * To achieve this, you need to use the methods in the Extractor class 
	 * @param lines lines
	 */
	public static void populateWordToSoundMap(List<String> lines) {	
		// YOUR CODE GOES HERE
		String word,sound;
		for(int i = 0; i < lines.size(); i++){
			word = Extractor.extractWordFromLine(lines.get(i));
			sound = Extractor.extractSoundFromLine(lines.get(i));
			wordToSound.put(word,sound);
		}
	}
	
	/**
	 * Given a list of all entries in the database, this method populates the 
	 * soundGroupToSimilarWords map as follows: the key is the sound-group, 
	 * and the value is a BST containing all the words that share that sound-group. 
	 * For example, if the line entry is "moderated M AA1 D ER0 EY2 T IH0 D", the key would 
	 * be "EY2 T IH0 D" and the value would be a BST containing "moderated" and all other
	 * words in the database that share the sound-group "EY2 T IH0 D"
	 * To achieve this, you need to use the methods in the Extractor class
	 * @param lines content of the database
	 */
	public static void populateSoundGroupToSimilarWordsMap(List<String> lines) {
		// YOUR CODE GOES HERE
		String word,soundgroup;
		for(int i = 0; i < lines.size(); i++){
			soundgroup = Extractor.extractSoundGroupFromSound(lines.get(i)); //grab soundgroup and word associated with soundgroup
			word = Extractor.extractWordFromLine(lines.get(i));
			if(soundGroupToSimilarWords.get(soundgroup) == null){ //if soundgroup isnt in map yet
				BST<String> soundgroupTree = new BST<>();
				soundgroupTree.insert(word);
				soundGroupToSimilarWords.put(soundgroup,soundgroupTree); //adds soundgroup and BST to map
			}
			else{
				soundGroupToSimilarWords.get(soundgroup).insert(word); //grab the BST associated with soundgroup and add word to BST
			}
		}
	}
	/**
	 * soundgroupSimilarWordsInList map to help me store BSTs in a smaller contained manner for multi input args for quicker search.
     */
	private static Map<String, BST<String>> soundgroupSimilarWordsInList;
	/**
	 * checker map to help me know if a word is a duplicate, new entry,non first instant or unrecognized by using code numbers of 4,0,1 and 2.
     */
	private static Map<String, Integer> checker;
	/**
	 * Given a list of words, e.g., [word1, word2, word3, word4], this method checks whether 
	 * word1 is similar to word2, word3, and word4. Then checks whether word2 is similar 
	 * to word3 and word4, and finally whether word3 is similar to word4.
	 *
	 * <p>For example if the list contains: [calculated legislated hello world miscalleneous 
	 * miscalculated encapsulated LIBERATED Sophisticated perculated hello], 
	 * the output should exactly be as follows:
	 *
	 * <p>"calculated" sounds similar to: "legislated"
	 *	"hello" sounds similar to: none
	 *	"world" sounds similar to: none
	 *	"miscalculated" sounds similar to: "encapsulated" "LIBERATED" "Sophisticated"
	 *	Unrecognized words: "miscalleneous" "perculated"
     *
     * 	<p>Note however that: 
	 * a) if a word was already found similar, then it will be ignored hereafter
	 * b) the behavior is case insensitive
	 * c) the subsequent occurrence of a given word is ignored  
	 * d) words that couldn’t be found in the database are deemed unrecognizable 
	 * e) words are displayed within quotes
	 * @param words list of words to examine
	 */
	public static void findSimilarWordsInList(String words[]) {
		// YOUR CODE GOES HERE
		soundgroupSimilarWordsInList = new java.util.HashMap<>();
		checker = new java.util.HashMap<>();
		int tracker1 = 0;
		int tracker2 = 0;
		int[] update1 =  new int[words.length];
		String sound,word,soundGroup,soundandword,soundGroup2,similar,holder;
		String[] sounds = new String[words.length];
		while(tracker1 < words.length){ //figures out words soundgroup and then inserts into BST
			word = words[tracker1]; //extract the word from args , sound (by using word to find sound via the wordToSound map), and soundGroup (by using the Extractor program to get soundGroup from word
			sound = wordToSound.get(word.toUpperCase());
			if(sound == null){ //this would help detect unrecognizable words
				soundGroup = "";
			}
			else{ //get the sound group
				soundGroup = Extractor.extractSoundGroupFromSound(sound);
			}
			if(wordToSound.get(word.toUpperCase()) == null){ //if word isnt in .txt file 
				checker.put(word,2); //if put 2, then that means the word is unrecognized
			}
			else if(soundgroupSimilarWordsInList.get(soundGroup) == null){ //if soundgroup isnt in map yet
				BST<String> soundgroupTree = new BST<>();
				soundgroupTree.insert(word);
				soundgroupSimilarWordsInList.put(soundGroup,soundgroupTree); //adds soundgroup and BST to map
				checker.put(word,0); //if put 0, then that means that is the first instant of that sound group in the tree
			}
			else{
				// soundgroupSimilarWordsInList.get(soundGroup).insert(word); //grab the BST associated with soundgroup and add word to BST
				// checker.put(word,1); //if put 1, then that means that is the non first instant of that sound group in the tree
				String check = "\""+word.toUpperCase()+"\"";; //was +"hello"+ also was +word+
				//System.out.println(soundgroupSimilarWordsInList.get(soundGroup).toString());
				//if(check.equals(soundgroupSimilarWordsInList.get(soundGroup).toString().toUpperCase()) == true){ //handles the case where a duplicate word is seen in the argument
				//System.out.println(soundgroupSimilarWordsInList.get(soundGroup).toString()+"Tree");
				//System.out.println(check+"word capitalized");
				if(soundgroupSimilarWordsInList.get(soundGroup).toString().toUpperCase().contains(check) == true){ //to indicate duplicate , we do make the hashmap 0 but make the int array 4, I had to add an int array due to the program being able to tell the difference between hello hello vs hello HelLO 
					//System.out.println(soundgroupSimilarWordsInList.get(soundGroup).toString()+"Tree");
					checker.put(word,0);
					update1[tracker1] = 4;
				}
				else{
					soundgroupSimilarWordsInList.get(soundGroup).insert(word); //grab the BST associated with soundgroup and add word to BST
					checker.put(word,1); //if put 1, then that means that is the non first instant of that sound group in the tree
				}
			}
			tracker1++;
		}
		//System.out.println(checker.get("hello"));
		while(tracker2 < words.length){
			similar = "";
			holder = "";
			String postquotes = "";
			int found = 1; //turn to 0 if first occurence of string was found
			if(checker.get(words[tracker2]) == 0 && update1[tracker2] != 4){ //do inner if statement if only non similar to anything else, use tokenizer or split to get none output 
				soundGroup2 = Extractor.extractSoundGroupFromSound(wordToSound.get(words[tracker2].toUpperCase()));
				similar = "\""+words[tracker2]+"\""+" sounds similar to: ";
				holder = soundgroupSimilarWordsInList.get(soundGroup2).toString();
				String[] holder2 = holder.split(" "); //responsible for putting words with commas into final string
				postquotes = holder.replaceAll("\"", "");
				String[] split = postquotes.split(" "); //to get a proper comparision of two string arrays with words but without quotes, final culmination of split,holder and postquotes strings
				for(int y = 0; y < words.length; y++){ //top for loop looks at each word in the methods input parameter
					found =1;
					for(int z = 0; z < split.length; z++){//inner loop compares values available from BST of similiar sounds words and sees which value shows up first in the methods input parameter etc
						if(split[z].equals(words[tracker2])==true){ //ignores the first word of similiar sounding words
						}
						else if(found == 0){ //dont do anything if we found the value
						}
						else if(split[z].equals(words[y])==true){ //orders the words
							if(similar.contains(words[y]) == true){ //check if our string already has the value in it
								
							}
							else{  //if not equal to anything in our string , add it to the string
								similar = similar+holder2[z]+" ";
								found = 0;
							}
						}
						else{
						}
					}
				}
				//System.out.println(similar);
				if(similar.equals("\""+words[tracker2]+"\""+" sounds similar to: ") == true){
					similar = similar+"none";
					System.out.println(similar);
				}
				else{
					System.out.println(similar.trim()); //just added .trim()
				}
				String check = "\""+words[tracker2]+"\"";
				if(check.equals(soundgroupSimilarWordsInList.get(soundGroup2).toString()) == true){ //handles the case where a duplicate word is seen in the argument
					checker.put(words[tracker2],3); //when 3 is put, this is only to ensure 
				}
			}
			tracker2++;
		}
		tracker2 = 0;
		String unrecognized = "Unrecognized words: ";
		while(tracker2 < words.length){
			if(checker.get(words[tracker2]) == 2){ //unrecognized 
				unrecognized = unrecognized+"\""+words[tracker2]+"\""+" ";
			}
			tracker2++;
		}
		if(unrecognized.length() == 20){ //if the unrecognized string was left untouched
			System.out.println("Unrecognized words: none");
		}
		else{
			System.out.println(unrecognized.trim()); //just added .trim()
		}
		// System.out.println(checker.get("hello"));
		// System.out.println(wordToSound.get("HELLO"));
		// System.out.println(Extractor.extractSoundGroupFromSound("HH AH0 L OW1"));
		// System.out.println(soundgroupSimilarWordsInList.get("EY2 T AH0 D"));
		// System.out.println(soundgroupSimilarWordsInList.get("EY2 T IH0 D").toString());
		// System.out.println(soundgroupSimilarWordsInList.get("AW1 N D Z"));
		// System.out.println(soundgroupSimilarWordsInList.get("ER1 L D"));
		// System.out.println(wordToSound.get(words[10].toUpperCase()));
		// System.out.println(words[10]);
		// String check = "\""+"hello"+"\"";
		// System.out.println(check);
		// System.out.println(soundgroupSimilarWordsInList.get("OW1")+"hi");
		// System.out.println(check.equals(soundgroupSimilarWordsInList.get("OW1").toString()));
		// System.out.println(soundgroupSimilarWordsInList.get("EY2 T IH0 D").find("false"));
		// System.out.println(soundgroupSimilarWordsInList.get("EY2 T IH0 D").find("Sophisticated"));
	}

	/**
	 *Given a passed word this method prints all similarly sounding words in ascending order (including the passed word)
	 * For example:	java SimilarSounds dimension
	 * Words similar to "dimension": "ASCENSION" "ATTENTION" "CONTENTION" "CONVENTION" "DECLENSION"
	 * "DETENTION" "DIMENSION" "DISSENSION" "EXTENSION" "GENTIAN" "HENSCHEN" "LAURENTIAN"
	 * "MENTION" "PENSION" "PRETENSION" "PREVENTION" "RETENTION" "SUSPENSION" "TENSION"
     *
	 * <p>Note how the word passed as an argument must still appear in the output. 
	 * However, if it cannot be found in the database an appropriate error message should be displayed
	 * @param theWord word to process
	 */
	public static void findSimilarWordsTo(String theWord) {		
		// YOUR CODE GOES HERE
		String sound,soundGroup;
		sound = wordToSound.get(theWord.toUpperCase());
		if(sound == null){
			soundGroup = "Unrecognized word: "+"\""+theWord+"\"";
			System.out.println(soundGroup);	
		}
		else{
			soundGroup = Extractor.extractSoundGroupFromSound(sound);
			System.out.println("Words similar to "+"\""+theWord+"\""+": "+soundGroupToSimilarWords.get(soundGroup).toString());
		}
	}
}
