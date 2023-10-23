/**
 *  This class implements a BST.
 *  
 *  @param <T> the type of the key.
 *
 *  @author W. Masri and Javier Ordonez
 */
class BST<T extends Comparable<T>> {
	// **************//
	// DO NO CHANGE
	
	/**
	 *  Node class.
	 *  @param <T> the type of the key.
	 */
	class Node<T extends Comparable<T>> 
	{
		/**
		*  key that uniquely identifies the node.
		*/
		T key;
		/**
		*  references to the left and right nodes.
		*/
		Node<T> left, right;
		/**
		 *  Constructor that forms a node based on the input it is given.
	     *  @param item is the value we want this node to hold
		 */
		public Node(T item) {  
			key = item; 
			left = right = null; 
		}
		/**
		 *  This is the toString method in charge of returning the value of the node.
	     *  @return is the key
	     */
		public String toString() { 
			return "" + key; 
		}
	}
	/**
	 *  The root of the BST.
	 */
	Node<T> root;
	/**
	 *  This is the root node we should use to begin the BST.
	 */
	public BST() { 
		root = null; 
	}
	/**
	 *  This is the toString method in charge of returning a string value that holds the inorder call of the BST.
	 *  @return is the string that holds the inorder iteration of the tree
	 */
	public String toString() { 
		return inorderToString(); 
	}
	// DO NO CHANGE
	// **************//
	
	
	/**
	 *  This method returns a string in which the elements are listed in an inorder fashion. 
	 *  Your implementation must be recursive.
	 *  Note: you can create private helper methods
	 *  @return string in which the elements are listed in an inorder fashion
	 */
	public String inorderToString() {
		// YOUR CODE GOES HERE
		//String inorder = "";
		if(root == null){
			return "";
		}
		inorder = toStringHelp(root);
		// System.out.println(inorder.length());
		// System.out.println(inorder);
		inorder = inorder.substring(0, inorder.length() -1); //this removes the extra space at the end that blocked the yay tests to be successful
		String inorderactual = ""+inorder; //to allow the string to clear and be ready for the next call
		inorder = "";
		return inorderactual;
	}
	/**
	 *  The variable inorder was placed to help remove some issues I had with traversing
	 *  and taking account of inorder recording spots. Essentially holds the inorder traversal.
	 */
	private String inorder = "";
	/**
	 *  This method helps the original method actually perform recursion for the 
	 *  inorder search given that the original method does not take in method arguments. 
	 *  @param node is the root node to help us traverse through the BST
	 *  @return inorder is the string where all the values are stored through a inorder traverse
	 */
	private String toStringHelp(Node<T> node){ //was void
		if(node == null){
			return "";
		}
		toStringHelp(node.left);
		inorder = inorder+"\""+node.key+"\""+" ";
		//System.out.println(inorder);
		toStringHelp(node.right);
		return inorder;
	}
	/**
	 *  This method inserts a node in the BST. You can implement it iteratively or recursively.
	 *  Note: you can create private helper methods
	 *  @param key to insert
	 */
	public void insert(T key) {
		// YOUR CODE GOES HERE
		//Node<T> newNode = new Node<>(key);
		// Node<T> currNode = root;
		//if(root == null){
		//	root = newNode;
		//}
		//else{
		root = insertHelp(root,key);
		//}	
	}
	/**
	 *  This method is a helper method for the insert method that allows us to use 
	 *  recursion but one that allows us to use the .left and .right
	 *  pointers that weren't possible with the original method. 
	 *  @param node is the root of the tree we would like to traverse from
	 *  @param key is the generic input we want to insert
	 *  @return is the node we are currently looking at which really, is never used
	 */
	private Node<T> insertHelp(Node<T> node, T key){
		if(node == null){
			Node<T> newNode = new Node<>(key);
			node = newNode;
			return node;
		}  
		if(key.compareTo(node.key) < 0){ //look to the left
			node.left = insertHelp(node.left,key);
		}
		else if(key.compareTo(node.key) > 0){ //look to the right
			node.right = insertHelp(node.right,key);
		}
		return node;
	}
	/**
	 *  This method finds and returns a node in the BST. You can implement it iteratively or recursively.
	 *  It should return null if not match is found.
	 *  Note: you can create private helper methods
	 *  @param key to find
	 *  @return the node associated with the key if found
	 */
	public Node<T> find(T key)	{ 	//to get the toString to work comment out this and the helper code				
		//YOUR CODE GOES HERE
		//System.out.println(findHelp(root,key));
		if(findHelp(root,key) != null){
			Node<T> holder = new Node<>(key); //just added
			//System.out.println(found);
			found = findHelp(root,key); //was null
			holder = found; //just added
			found = null; //just added
			return holder; //was root
		}
		else{
			return null;
		}
	}
	/**
	 *  The found node is the node used to keep track of the node if found or not.
	 */
	private Node<T> found;
	/**
	 *  This method finds and returns a node in the BST. You can implement it iteratively or recursively.
	 *  It should return null if not match is found.
	 *  @param key to find
	 *  @param node is the node we are starting the traverse from
	 *  @return the node associated with the key if found
	 */
	private Node<T> findHelp(Node<T> node, T key){
		if(node == null){
			//System.out.println("Here 1");
			return null;
		}
		if(key.compareTo(node.key) == 0){
			//System.out.println("Here 2");
			Node<T> newNode2 = new Node<>(key);
			found = newNode2; //was node
			return found;
		}
		else if(found != null){ //check if found node was tampered with already, if yes, start exiting recursion
			//System.out.println("Here 3");
			return found;
		}
		else{
			//System.out.println("Here 4");
			if(key.compareTo(node.key) < 0){
				//System.out.println("Here 4");
				found = findHelp(node.left,key);
			}
			else if(key.compareTo(node.key) > 0){
				//System.out.println("Here 5");
				found = findHelp(node.right,key);
			}
		}
		return found;
	}

	/**
	 *  Main Method For Your Testing -- Edit all you want.
	 *  
	 *  @param args not used
	 */
	public static void main(String[] args) {
		/*
							 50
						  /	      \
						30    	  70
	                 /     \    /     \
	                20     40  60     80   
		*/
		
		
		BST<Integer> tree1 = new BST<>();
		tree1.insert(50); tree1.insert(30); tree1.insert(20); tree1.insert(40);
		tree1.insert(70); tree1.insert(60); tree1.insert(80);
		// System.out.println(tree1.root.key);
		// System.out.println(tree1.root.left.key);
		// System.out.println(tree1.root.right.key);
		//System.out.println(tree1.toString());
		if (tree1.find(70) != null) {
			System.out.println("Yay1");
		}
		// System.out.println(tree1.find(70));
		//System.out.println(tree1.toString());
		if (tree1.find(90) == null) {
			System.out.println("Yay2");
		}
		// System.out.println(tree1.find(90));
		if (tree1.find(100) == null) {
			System.out.println("Yay10");
		}
		// System.out.println(tree1.find(100));
		if (tree1.find(40) != null) {
			System.out.println("Yay11");
		}
		// System.out.println(tree1.find(40));
		//System.out.println(tree1.toString());
		if (tree1.toString().equals("\"20\" \"30\" \"40\" \"50\" \"60\" \"70\" \"80\"") == true) {
			System.out.println("Yay3");
		}
		
		
		BST<String> tree2 = new BST<>();
		tree2.insert("50"); tree2.insert("30"); tree2.insert("20"); tree2.insert("40");
		tree2.insert("70"); tree2.insert("60"); tree2.insert("80");
		
		if (tree2.find("70") != null) {
			System.out.println("Yay4");
		}
		// System.out.println(tree2.find("70"));
		if (tree2.find("90") == null) {
			System.out.println("Yay5");
		}
		// System.out.println(tree2.find("90"));
		if (tree2.toString().equals("\"20\" \"30\" \"40\" \"50\" \"60\" \"70\" \"80\"") == true) {
			System.out.println("Yay6");
		}
	}
	
}
