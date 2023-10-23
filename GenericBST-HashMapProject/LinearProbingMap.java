import java.util.Set;
import java.util.Collection;
import java.util.ArrayList;
import java.util.List;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * No need for description.
 * @param <K> no need for description.
 * @param <V> no need for description.
 */	
public class LinearProbingMap<K, V> implements Map<K, V>
{
	/**
	 * No need for description.
	 * @param <K> no need for description.
	 * @param <V> no need for description.
	 */	
	class Pair<K,V> {
		/**
		* No need for a description.
		*/
		private K key;
		/**
		* No need for a description.
		*/	
		private V value;
		/**
		* No need for a description.
		* @param key no need for a description
		* @param value no need for a description
		*/	
		public Pair(K key, V value){
			this.key = key;
			this.value = value;
		}
		/**
		* No need for a description.
		*/	
		public Pair(){
			this.key = null;
			this.value = null;
		}
		public K getKey(){ return key; }
		public V getValue(){ return value; }
		public void setKey(K key){ this.key = key; }
		public void setValue(V value){ this.value = value; }
		@Override public int hashCode() {  
			return key.hashCode(); 
		}
		@Override public boolean equals(Object obj) {  
			if (obj == null) return false;
			if (!(obj instanceof Pair)) return false;
			Pair pair = (Pair)obj;
			return pair.key.equals(key); 
		}
	}

	/**
	 * No need for a description.
	 */	
	private static final int DEFAULT_CAPACITY = 400000;
	/**
	 * No need for a description.
	 */	
	private int size;
	/**
	 * No need for a description.
	 */	
	private int capacity; 
	/**
	 * No need for a description.
	 */	
	private Pair<K, V>[] table;
	/**
	 * No need for a description.
	 */	
	private Pair<K, V> tombstone;  // has no impact since we are not implementing remove()
	
	/**
	 * No need for a description.
	 * @param capacity no need for a description.
	 */	
	@SuppressWarnings("unchecked")
	public LinearProbingMap(int capacity)
	{
		this.capacity = capacity;
		size = 0;
		table = (Pair<K, V>[])new Pair[capacity];
		tombstone = (Pair<K, V>)new Pair();	// has no impact since we are not implementing remove()
	}
	
	/**
	 * No need for a description.
	 */	
	public LinearProbingMap() {
		this(DEFAULT_CAPACITY);
	}

	/**
	 * No need for a description.
	 * @return no need for a description.
	 */	
	public int size() { 
		return size; 
	}

	/**
	 * No need for a description.
	 * @return no need for a description.
	 */		
	public boolean isEmpty() { 
		return size == 0; 
	}

	/**
	 * No need for a description.
	 */	
	public void clear() {
		size = 0;
		for (int i = 0; i < capacity; i++) {
			table[i] = null;
		}
	}

	////////////*****ADD YOUR CODE HERE******////////////////////

	/**
	 * This method gets values from the HashMap based on if it is in its spot, can't be found (null entry) or has been found somewhere. 
	 * @param key the key is the generic data type which identifies what to turn into Hashcode for quick search up.
	 * @return can be either null or the value based on the conditions said above. If found (table[hash].value versus not found (null). 
	 */			
	public V get(Object key) {
		// YOUR CODE GOES HERE
		int hash;
		hash = computeHash(key);
		if(table[hash] == null){ //if map does not have the key now works to check if the hashvalue at that spot is empty if nothing is there
			return null;
		}
		else{
			int checker = 0;
			while(checker != 1){
				if(table[hash] == null){ //if we kept moving along the hashmap but the key was never found
					return null;
				}
				else if(table[hash].key.equals(key)){ // to check for duplicate key values
					checker = 1;
					return table[hash].value;
				}
				hash++;
			}
		}
		return null;
	}
	
	/**
	 * This puts the entry into the Map based on its Hashcode and whether or not something already occupies that initial spot.
	 * If the initial spot is already in use, it would linearly probe throughout the rest of the map until, an empty spot is  
	 * available or until the case a duplicate matching key is found. If the spot is available simply insert or if the duplicate is 
	 * found, replace the old existing term with the newer value of the entry. 
	 * @param key is what we would turn into Hash code to determine where to place it in the Hash Map based on scenarios.
	 * @param value is the value of the entry we would like to store at that specific location in the Hash Map. 
	 * @return could either be the new Value of the HashMap or the old duplicate value of the Hashmap if a duplicate was found. 
	 */	
	public V put(K key, V value) {  
		// YOUR CODE GOES HERE
		int hash;
		hash = computeHash(key);
		Pair<K,V> holder = new Pair<>(key,value);
		int checker = 0;
		while(checker != 1){
			if (table[hash] == null){ //check if that current spot is completely free
				table[hash] = holder;
				checker = 1;
				size++;
			}	
			else if (table[hash] != null) { //to check for duplicates/over populated hashmap/no actual occurences of entry with hash code matching the key we have
				if(table[hash].key.equals(key)){ // to check for duplicate key values
					V valueHolder = table[hash].value; //just added
					table[hash] = holder;
					checker = 1;
					//size++;
					return valueHolder; //just added
				}
				else if(table[hash] == null){
					table[hash] = holder;
					checker = 1;
					size++;
					return table[hash].value; //just added
				}
			}
			hash++;
		}
		return null;
	}

	////////////***********////////////////////

	/**
	 * No need for a description.
	 * @param key no need for a description.
	 * @return no need for a description.
	 */	
	public V remove(Object key) {
		return null; // DO NOT IMPLEMENT
	}
	
	/**
	 * No need for a description.
	 * @param key no need for a description.
	 * @return no need for a description.
	 */	
	private int computeHash(Object key)
	{
		int hash = Math.abs(key.hashCode()) % capacity;
		return hash;
	}

	/**
	 * No need for a description.
	 * @return no need for a description.
	 */	
	public String toString()
	{
		StringBuilder st = new StringBuilder();
		for (int i = 0; i < capacity; i++) {
			if (table[i] != null) {
				st.append("(" + table[i].key + ", " + table[i].value + ")");
			}
		}
		return st.toString();
	}

	/**
	 * No need for a description.
	 * @param key no need for a description.
	 * @return no need for a description.
	 */	
	public boolean containsKey(Object key) {
		for (int i = 0; i < capacity; i++) {
			if (table[i] != tombstone && table[i] != null) {
				if (table[i].key.equals(key)) {
					return true;
				}
			}
		}
		return false;
	}
	
	/**
	 * No need for a description.
	 * @param value no need for a description.
	 * @return no need for a description.
	 */	
	public boolean containsValue(Object value) {
		for (int i = 0; i < capacity; i++) {
			if (table[i] != tombstone && table[i] != null) {
				if (table[i].value.equals(value)) {
					return true;
				}
			}
		}
		return false;
	}
	
	/**
	 * No need for a description.
	 * @return no need for a description.
	 */	
	public Set<K> keySet() {
		HashSet<K> set = new HashSet<K>();
		for (int i = 0; i < capacity; i++) {
			if (table[i] != tombstone && table[i] != null) {
				set.add(table[i].key);
			}
		}
		return set;
	}	
	
	/**
	 * No need for a description.
	 * @return no need for a description.
	 */	
	public Collection<V> values() {
		ArrayList<V> list = new ArrayList<V>();
		for (int i = 0; i < capacity; i++) {
			if (table[i] != tombstone && table[i] != null) {
				list.add(table[i].value);
			}
		}
		return list;
	}

	/**
	 * No need for a description.
	 * @return no need for a description.
	 */		
	public Set<Map.Entry<K,V>>	entrySet() {
		return null;
	}

	/**
	 * No need for a description.
	 * @param m no need for a description.
	 */	
	public void putAll(Map<? extends K,? extends V> m) {
	}
	
	/**
	 *  Main Method For Your Testing -- Edit all you want.
	 *  
	 *  @param args not used
	 */
	public static void main(String[] args)
	{
		int n = 10;
		LinearProbingMap<String, Integer> dict = new LinearProbingMap<>(n*2);
		
		for (int i = 1; i <= n; i++) {
			dict.put(""+i, i);
		}
		if (dict.size() == 10) {
			System.out.println("Yay1");
		}
		if (dict.get("5").equals(5)) {
			System.out.println("Yay2");
		}
		
		dict.put("6", 60);
		dict.put("10", 100);
		dict.put("20", 200);
		if (dict.get("6") == 60) {
			System.out.println("Yay3");
		}
		if (dict.get("10") == 100) {
			System.out.println("Yay4");
		}
		if (dict.get("20") == 200) {
			System.out.println("Yay5");
		}
		if (dict.size() == 11) {
			System.out.println("Yay6");
		}
		
		if (dict.get("200") == null) {
			System.out.println("Yay7");
		}	
	}
}
