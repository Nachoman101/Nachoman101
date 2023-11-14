/* 
 * stoplight.c
 *
 * 31-1-2003 : GWA : Stub functions created for CS161 Asst1.
 *
 * NB: You can use any synchronization primitives available to solve
 * the stoplight problem in this file.
 */


/*
 * 
 * Includes
 *
 */

#include <types.h>
#include <lib.h>
#include <test.h>
#include <thread.h>
#include <synch.h>
#include <machine/spl.h>


/*
 *
 * Constants
 *
 */

/*
 * Number of vehicles created.
 */

#define NVEHICLES 20

/*
 * Lock Declarations we need for the intersection(One Lock for each intersection chunk, 3 locks). If you hold a lock, you are in this part of the intersection!!!
 */
struct lock *AB;
struct lock *BC;
struct lock *CA;
//AB = lock_create("AB");
//BC = lock_create("BC");
//CA = lock_create("CA");
/*
 *  We also need to keep track of the number of vehicles there are per lane, think of this
 *  As the "true" critical section. Checks the number of vehicles in this lane, this will 
 *  will be used to determine if a truck can go or not.
 */
int vehiclesInLaneA;
int vehiclesInLaneB;
int vehiclesInLaneC;
struct lock *mutexVehiclesInLaneA;
struct lock *mutexVehiclesInLaneB;
struct lock *mutexVehiclesInLaneC;
//mutexVehiclesInLaneA = lock_create("mutexVehiclesInLaneA");
//mutexVehiclesInLaneB = lock_create("mutexVehiclesInLaneB");
//mutexVehiclesInLaneC = lock_create("mutexVehiclesInLaneC");
/*
 *  We will also need one more lock, this is the left turn lock. This lock will determine
 *  if a vehicle will be able to make a left turn or not. There will also be a variable
 *  involved with this lock that will determine if there are two vehicles already trying to 
 *  make a left. If two vehicles are trying to make a left, put the other vehicles trying 
 *  to make a left to sleep
 */
int vehiclesMakingLeftTurn;
struct lock *mutexCheckVehiclesMakingLeftTurn;
//mutexCheckVehiclesMakingLeftTurn = lock_create("mutexCheckVehiclesMakingLeftTurn");
/*
 *  Now finally, we need to be able to have two locations to put threads to sleep. The
 *  first location would be a location to put trucks to sleep, in this case, this
 *  would be three different locations. The second location would be where we put the 
 *  vehicles making a left turn to sleep.  
 */
const void *trucksSleepingAtLaneA;
const void *trucksSleepingAtLaneB;
const void *trucksSleepingAtLaneC;
const void *trucksSleepingAtAllLanes;
const void *sleepingVehiclesMakingLeftTurn;
/*
 * It occured to me that you probably want to make the main function sleep 
 * and let the other threads finish their stuff first before letting the main exit
 */
int finishedProcesses;
struct lock *mutexMainProcessToExit;
//mutexMainProcessToExit = lock_create("mutexMainProcessToExit");
const void *mainProcessSleepingHere;


/*
 *
 * Function Definitions
 *
 */



/*
 * turnleft()
 *
 * Arguments:
 *      unsigned long vehicledirection: the direction from which the vehicle
 *              approaches the intersection.
 *      unsigned long vehiclenumber: the vehicle id number for printing purposes.
 *
 * Returns:
 *      nothing.
 *
 * Notes:
 *      This function should implement making a left turn through the 
 *      intersection from any direction.
 *      Write and comment this function.
 */

static
void
turnleft(unsigned long vehicledirection,
		unsigned long vehiclenumber,
		unsigned long vehicletype)
{
	/*
	 * Avoid unused variable warnings.
	 */

	(void) vehicledirection;
	(void) vehiclenumber;
	(void) vehicletype;
	if(vehicletype == 0){ //Vehicle is a Car
		if(vehicledirection == 0){ //Coming From Route A - Needs Locks AB-BC
			lock_acquire(AB);
			lock_acquire(mutexVehiclesInLaneA);
			vehiclesInLaneA--;
			//kprintf("vehiclesInLaneA: %d\n",vehiclesInLaneA);
			if(vehiclesInLaneA == 0){
				lock_release(mutexVehiclesInLaneA);
				//kprintf("Vehicle %ld is waking up trucksleepingatLaneA\n",vehiclenumber);
				int spl = splhigh();
				thread_wakeup(trucksSleepingAtAllLanes);
				splx(spl); 
			}
			else{
				lock_release(mutexVehiclesInLaneA);
			}
			kprintf("\nCar # %ld is entering the Intersection from Route A to make a left turn, is holding lock AB. Attempting to grab lock BC.\n",vehiclenumber);
			lock_acquire(BC);
			//kprintf("Here1\n");
			lock_release(AB);
			kprintf("\nCar # %ld is entering the Intersection from Route A to make a left turn, is holding lock BC. Will now finish left turn from Route A.\n",vehiclenumber);
			lock_release(BC);
			lock_acquire(mutexCheckVehiclesMakingLeftTurn);
			vehiclesMakingLeftTurn--;
			//kprintf("vehiclesMakingLeftTurn %d\n",vehiclesMakingLeftTurn);
			lock_release(mutexCheckVehiclesMakingLeftTurn);
			int spl = splhigh();
			thread_wakeup(sleepingVehiclesMakingLeftTurn);
			splx(spl);
		}
		else if(vehicledirection == 1){ //Coming from Route B - Needs Locks BC-CA
			lock_acquire(BC);
			lock_acquire(mutexVehiclesInLaneB);
			vehiclesInLaneB--;
			//kprintf("vehiclesInLaneB: %d\n",vehiclesInLaneB);
			if(vehiclesInLaneB == 0){
				lock_release(mutexVehiclesInLaneB);
				//kprintf("Vehicle %ld is waking up trucksleepingatLaneB\n",vehiclenumber);
				int spl = splhigh();
				thread_wakeup(trucksSleepingAtAllLanes);
				splx(spl); 
			}
			else{
				lock_release(mutexVehiclesInLaneB);
			}
			kprintf("\nCar # %ld is entering the Intersection from Route B to make a left turn, is holding lock BC. Attempting to grab lock CA.\n",vehiclenumber);
			lock_acquire(CA);
			//kprintf("Here2\n");
			lock_release(BC);
			kprintf("\nCar # %ld is entering the Intersection from Route B to make a left turn, is holding lock CA. Will now finish left turn from Route B.\n",vehiclenumber);
			lock_release(CA);
			lock_acquire(mutexCheckVehiclesMakingLeftTurn);
			vehiclesMakingLeftTurn--;
			//kprintf("vehiclesMakingLeftTurn: %d\n",vehiclesMakingLeftTurn);
			lock_release(mutexCheckVehiclesMakingLeftTurn);
			int spl = splhigh();
			thread_wakeup(sleepingVehiclesMakingLeftTurn);
			splx(spl);
		}
		else{  //Coming from Route C - Needs Locks CA-AB
			lock_acquire(CA);
			lock_acquire(mutexVehiclesInLaneC);
			vehiclesInLaneC--;
			//kprintf("vehiclesInLaneC: %d\n",vehiclesInLaneC);
			if(vehiclesInLaneC == 0){
				lock_release(mutexVehiclesInLaneC);
				//kprintf("Vehicle %ld is waking up trucksleepingatLaneC\n",vehiclenumber);
				int spl = splhigh();
				thread_wakeup(trucksSleepingAtAllLanes);
				splx(spl); 
			}
			else{
				lock_release(mutexVehiclesInLaneC);
			}
			kprintf("\nCar # %ld is entering the Intersection from Route C to make a left turn, is holding lock CA. Attempting to grab lock AB.\n",vehiclenumber);
			lock_acquire(AB);
			//kprintf("Here3\n");
			lock_release(CA);
			kprintf("\nCar # %ld is entering the Intersection from Route C to make a left turn, is holding lock AB. Will now finish left turn from Route C.\n",vehiclenumber);
			lock_release(AB);
			lock_acquire(mutexCheckVehiclesMakingLeftTurn);
			vehiclesMakingLeftTurn--;
			//kprintf("vehiclesMakingLeftTurn: %d\n",vehiclesMakingLeftTurn);
			lock_release(mutexCheckVehiclesMakingLeftTurn);
			int spl = splhigh();
			thread_wakeup(sleepingVehiclesMakingLeftTurn);
			splx(spl);
		}
	}
	else{ //Vehicle is a truck
		if(vehicledirection == 0){ //Coming From Route A - Needs Locks AB-BC
			lock_acquire(AB);
			kprintf("\nTruck # %ld is entering the Intersection from Route A to make a left turn, is holding lock AB. Attempting to grab lock BC.\n",vehiclenumber);
			lock_acquire(BC);
			//kprintf("Here4\n");
			lock_release(AB);
			kprintf("\nTruck # %ld is entering the Intersection from Route A to make a left turn, is holding lock BC. Will now finish left turn from Route A.\n",vehiclenumber);
			lock_release(BC);
			lock_acquire(mutexCheckVehiclesMakingLeftTurn);
			vehiclesMakingLeftTurn--;
			//kprintf("vehiclesMakingLeftTurn: %d\n",vehiclesMakingLeftTurn);
			lock_release(mutexCheckVehiclesMakingLeftTurn);
			int spl = splhigh();
			thread_wakeup(sleepingVehiclesMakingLeftTurn);
			splx(spl);
		}
		else if(vehicledirection == 1){ //Coming from Route B - Needs Locks BC-CA
			lock_acquire(BC);
			kprintf("\nTruck # %ld is entering the Intersection from Route B to make a left turn, is holding lock BC. Attempting to grab lock CA.\n",vehiclenumber);
			lock_acquire(CA);
			//kprintf("Here5\n");
			lock_release(BC);
			kprintf("\nTruck # %ld is entering the Intersection from Route B to make a left turn, is holding lock CA. Will now finish left turn from Route B.\n",vehiclenumber);
			lock_release(CA);
			lock_acquire(mutexCheckVehiclesMakingLeftTurn);
			vehiclesMakingLeftTurn--;
			//kprintf("vehiclesMakingLeftTurn: %d\n",vehiclesMakingLeftTurn);
			lock_release(mutexCheckVehiclesMakingLeftTurn);
			int spl = splhigh();
			thread_wakeup(sleepingVehiclesMakingLeftTurn);
			splx(spl);
		}
		else{  //Coming from Route C - Needs Locks CA-AB
			lock_acquire(CA);
			kprintf("\nTruck # %ld is entering the Intersection from Route C to make a left turn, is holding lock CA. Attempting to grab lock AB.\n",vehiclenumber);
			lock_acquire(AB);
			//kprintf("Here6\n");
			lock_release(CA);
			kprintf("\nTruck # %ld is entering the Intersection from Route C to make a left turn, is holding lock AB. Will now finish left turn from Route C.\n",vehiclenumber);
			lock_release(AB);
			lock_acquire(mutexCheckVehiclesMakingLeftTurn);
			vehiclesMakingLeftTurn--;
			//kprintf("vehiclesMakingLeftTurn: %d\n",vehiclesMakingLeftTurn);
			lock_release(mutexCheckVehiclesMakingLeftTurn);
			int spl = splhigh();
			thread_wakeup(sleepingVehiclesMakingLeftTurn);
			splx(spl);
		}
	}
}


/*
 * turnright()
 *
 * Arguments:
 *      unsigned long vehicledirection: the direction from which the vehicle
 *              approaches the intersection.
 *      unsigned long vehiclenumber: the vehicle id number for printing purposes.
 *
 * Returns:
 *      nothing.
 *
 * Notes:
 *      This function should implement making a right turn through the 
 *      intersection from any direction.
 *      Write and comment this function.
 */

static
void
turnright(unsigned long vehicledirection,
		unsigned long vehiclenumber,
		unsigned long vehicletype)
{
	/*
	 * Avoid unused variable warnings.
	 */

	(void) vehicledirection;
	(void) vehiclenumber;
	(void) vehicletype;
	if(vehicletype == 0){ //Vehicle is a Car
		if(vehicledirection == 0){ //Coming From Route A - Needs Locks AB
			lock_acquire(AB);
			lock_acquire(mutexVehiclesInLaneA);
			vehiclesInLaneA--;
			//kprintf("vehiclesInLaneA: %d\n",vehiclesInLaneA);
			if(vehiclesInLaneA == 0){
				//kprintf("Vehicle %ld is waking up trucksleepingatLaneA\n",vehiclenumber);
				lock_release(mutexVehiclesInLaneA);
				int spl = splhigh();
				thread_wakeup(trucksSleepingAtAllLanes);
				splx(spl); 
			}
			else{
				lock_release(mutexVehiclesInLaneA);
			}
			kprintf("\nCar # %ld is entering the Intersection from Route A to make a right turn, is holding lock AB. Will now finish right turn from Route A.\n",vehiclenumber);
			lock_release(AB);
		}
		else if(vehicledirection == 1){ //Coming from Route B - Needs Locks BC
			lock_acquire(BC);
			lock_acquire(mutexVehiclesInLaneB);
			vehiclesInLaneB--;
			//kprintf("vehiclesInLaneB: %d\n",vehiclesInLaneB);
			if(vehiclesInLaneB == 0){
				//kprintf("Vehicle %ld is waking up trucksleepingatLaneB\n",vehiclenumber);
				lock_release(mutexVehiclesInLaneB);
				int spl = splhigh();
				thread_wakeup(trucksSleepingAtAllLanes);
				splx(spl); 
			}
			else{
				lock_release(mutexVehiclesInLaneB);
			}
			kprintf("\nCar # %ld is entering the Intersection from Route B to make a right turn, is holding lock BC. Will now finish right turn from Route B.\n",vehiclenumber);
			lock_release(BC);
		}
		else{  //Coming from Route C - Needs Locks CA
			lock_acquire(CA);
			lock_acquire(mutexVehiclesInLaneC);
			vehiclesInLaneC--;
			if(vehiclesInLaneC == 0){
				lock_release(mutexVehiclesInLaneC);
				//kprintf("Vehicle %ld is waking up trucksleepingatLaneC\n",vehiclenumber);
				//kprintf("vehiclesInLaneC: %d\n",vehiclesInLaneC);
				int spl = splhigh();
				thread_wakeup(trucksSleepingAtAllLanes);
				splx(spl); 
			}
			else{
				lock_release(mutexVehiclesInLaneC);
				//kprintf("vehiclesInLaneC: %d\n",vehiclesInLaneC);
			}
			kprintf("\nCar # %ld is entering the Intersection from Route C to make a right turn, is holding lock CA. WIll now finish right turn from Route C.\n",vehiclenumber);
			lock_release(CA);
		}
	}
	else{ //Vehicle is a truck
		if(vehicledirection == 0){ //Coming From Route A - Needs Locks AB
			lock_acquire(AB);
			/*lock_acquire(mutexVehiclesInLaneA); //Check to see if vehiclesInLaneA is still 0, if so, awake any other truck still sleeping
			if(vehiclesInLaneA == 0){
				lock_release(mutexVehiclesInLaneA);
				int spl = splhigh();
				thread_wakeup(trucksSleepingAtLaneA);
				splx(spl); 
			}
			else{
				lock_release(mutexVehiclesInLaneA);
			}
			*/
			kprintf("\nTruck # %ld is entering the Intersection from Route A to make a right turn, is holding lock AB. Will now finish right turn from Route A.\n",vehiclenumber);
			lock_release(AB);
		}
		else if(vehicledirection == 1){ //Coming from Route B - Needs Locks BC
			lock_acquire(BC);
			/*lock_acquire(mutexVehiclesInLaneB); //Check to see if vehiclesInLaneA is still 0, if so, awake any other truck still sleeping
			if(vehiclesInLaneB == 0){
				lock_release(mutexVehiclesInLaneB);
				int spl = splhigh();
				thread_wakeup(trucksSleepingAtLaneB);
				splx(spl); 
			}
			else{
				lock_release(mutexVehiclesInLaneB);
			}
			*/
			kprintf("\nTruck # %ld is entering the Intersection from Route B to make a right turn, is holding lock BC. Will now finish right turn from Route B.\n",vehiclenumber);
			lock_release(BC);
		}
		else{  //Coming from Route C - Needs Locks CA
			lock_acquire(CA);
			/*lock_acquire(mutexVehiclesInLaneC); //Check to see if vehiclesInLaneA is still 0, if so, awake any other truck still sleeping
			if(vehiclesInLaneC == 0){
				lock_release(mutexVehiclesInLaneC);
				int spl = splhigh();
				thread_wakeup(trucksSleepingAtLaneC);
				splx(spl); 
			}
			else{
				lock_release(mutexVehiclesInLaneC);
			}
			*/
			kprintf("\nTruck # %ld is entering the Intersection from Route C to make a right turn, is holding lock CA. Will now finish right turn from Route C.\n",vehiclenumber);
			lock_release(CA);
		}
	}
}


/*
 * approachintersection()
 *
 * Arguments: 
 *      void * unusedpointer: currently unused.
 *      unsigned long vehiclenumber: holds vehicle id number.
 *
 * Returns:
 *      nothing.
 *
 * Notes:
 *      Change this function as necessary to implement your solution. These
 *      threads are created by createvehicles().  Each one must choose a direction
 *      randomly, approach the intersection, choose a turn randomly, and then
 *      complete that turn.  The code to choose a direction randomly is
 *      provided, the rest is left to you to implement.  Making a turn
 *      or going straight should be done by calling one of the functions
 *      above.
 */

static
void
approachintersection(void * unusedpointer,
		unsigned long vehiclenumber)
{
	int vehicledirection, turndirection, vehicletype;

	/*
	 * Avoid unused variable and function warnings.
	 */

	(void) unusedpointer;
	(void) vehiclenumber;
	(void) turnleft;
	(void) turnright;

	/*
	 * vehicledirection is set randomly.
	 */
	vehicledirection = random() % 3;  //Vehicle Direction Key: 0 = From A , 1 = From B, 2 = From C
	turndirection = random() % 2; // Vehicle Direction Key: 0 = Make a Left Turn, 1 = Make a Right Turn
	//turndirection = 1; // Vehicle Direction Key: 0 = Make a Left Turn, 1 = Make a Right Turn
	vehicletype = random() % 2; // Vehicle Direction Key: 0 = Vehicle is a Car, 1 = Vehicle is a Truck
	//vehicletype = 1;
	//kprintf("Vehicle # %ld with direction # %d spawned in of type #  %d\n", vehiclenumber, vehicledirection, vehicletype);
	if (vehicletype == 0){ //Vehicle is a Car
		//kprintf("Vehicle # %ld is here1\n", vehiclenumber);
		if(vehicledirection == 0){ //Car coming from Direction A
			lock_acquire(mutexVehiclesInLaneA);
			vehiclesInLaneA++;
			lock_release(mutexVehiclesInLaneA);
			//kprintf("Vehicle # %ld is here1.1\n", vehiclenumber);
			//kprintf("vehiclesInLaneA: %d\n",vehiclesInLaneA);
			if(turndirection == 0){ //Car coming from Direction A wants to make a left, will use locks AB-BC
				kprintf("\nCar # %ld is approaching the Intersection from Route A to make a left turn (Needs Locks AB-BC).\n",vehiclenumber);
				while(1){
					lock_acquire(mutexCheckVehiclesMakingLeftTurn);
					if(vehiclesMakingLeftTurn == 2){
						lock_release(mutexCheckVehiclesMakingLeftTurn);
						int spl = splhigh();
						thread_sleep(sleepingVehiclesMakingLeftTurn);
						splx(spl);
					}
					else{
						vehiclesMakingLeftTurn++;
						lock_release(mutexCheckVehiclesMakingLeftTurn);
						//kprintf("vehiclesMakingLeftTurn: %d\n",vehiclesMakingLeftTurn);
						break;
					}
				}
				//kprintf("Vehicle # %ld is here1.2\n", vehiclenumber);
				turnleft(vehicledirection, vehiclenumber, vehicletype);
			}
			else{ //Car coming from Direction A wants to make a right, will use locks AB
				kprintf("\nCar # %ld is approaching the Intersection from Route A to make a right turn (Needs Locks AB).\n",vehiclenumber);
				turnright(vehicledirection,vehiclenumber,vehicletype);
			}
		}
		else if (vehicledirection == 1){ //Car coming from Direction B
			lock_acquire(mutexVehiclesInLaneB);
			vehiclesInLaneB++;
			lock_release(mutexVehiclesInLaneB);
			//kprintf("Vehicle # %ld is here1.3\n", vehiclenumber);
			//kprintf("vehiclesInLaneB: %d\n",vehiclesInLaneB);
			if(turndirection == 0){ //Car coming from Direction B wants to make a left, will use locks BC-CA
				kprintf("\nCar # %ld is approaching the Intersection from Route B to make a left turn (Needs Locks BC-CA).\n",vehiclenumber);
					while(1){
					lock_acquire(mutexCheckVehiclesMakingLeftTurn);
					if(vehiclesMakingLeftTurn == 2){
						lock_release(mutexCheckVehiclesMakingLeftTurn);
						int spl = splhigh();
						thread_sleep(sleepingVehiclesMakingLeftTurn);
						splx(spl);
					}
					else{
						vehiclesMakingLeftTurn++;
						lock_release(mutexCheckVehiclesMakingLeftTurn);
						//kprintf("vehiclesMakingLeftTurn: %d\n",vehiclesMakingLeftTurn);
						break;
					}
				}
				//kprintf("Vehicle # %ld is here1.4\n", vehiclenumber);
				turnleft(vehicledirection, vehiclenumber, vehicletype);
			}
			else{ //Car coming from Direction B wants to make a right, will use locks BC
				//kprintf("\nCar # %ld is approaching the Intersection from Route B to make a right turn (Needs Locks BC).\n",vehiclenumber);
				turnright(vehicledirection,vehiclenumber,vehicletype);
			}		
		}
		else{ //Car coming from Direction C
			lock_acquire(mutexVehiclesInLaneC);
			vehiclesInLaneC++;
			lock_release(mutexVehiclesInLaneC);
			//kprintf("Vehicle # %ld is here1.5\n", vehiclenumber);
			//kprintf("vehiclesInLaneC: %d\n",vehiclesInLaneC);
			if(turndirection == 0){ //Car coming from Direction C wants to make a left, will use locks CA-AB
				kprintf("\nCar # %ld is approaching the Intersection from Route C to make a left turn (Needs Locks CA-AB).\n",vehiclenumber);	 
				while(1){
					lock_acquire(mutexCheckVehiclesMakingLeftTurn);
					if(vehiclesMakingLeftTurn == 2){
						lock_release(mutexCheckVehiclesMakingLeftTurn);
						int spl = splhigh();
						thread_sleep(sleepingVehiclesMakingLeftTurn);
						splx(spl);
					}
					else{
						vehiclesMakingLeftTurn++;
						lock_release(mutexCheckVehiclesMakingLeftTurn);
						//kprintf("vehiclesMakingLeftTurn: %d\n",vehiclesMakingLeftTurn);
						break;
					}
				}
				//kprintf("Vehicle # %ld is here1.6\n", vehiclenumber);
				turnleft(vehicledirection, vehiclenumber, vehicletype);		
			}
			else{ //Car coming from Direction C wants to make a right, will use locks CA
				kprintf("\nCar # %ld is approaching the Intersection from Route C to make a right turn (Needs Locks CA).\n",vehiclenumber);	
				turnright(vehicledirection,vehiclenumber,vehicletype);
			}			
		}
	}
	else{ //Vehicle is a Truck
		//kprintf("Vehicle # %ld is here2\n", vehiclenumber);
		if(vehicledirection == 0){ //Car coming from Direction A
			while(1){ //Checks if any vehicle is in Lane A priot to trucks entrance
				lock_acquire(mutexVehiclesInLaneA);
				//kprintf("Vehicle # %ld has grabbed the mutexVehiclesInLaneA lock\n",vehiclenumber);
				if(vehiclesInLaneA == 0){
					lock_release(mutexVehiclesInLaneA);
					break;
				}
				else{
					lock_release(mutexVehiclesInLaneA);
					int spl = splhigh();
					thread_sleep(trucksSleepingAtAllLanes);
					splx(spl);
				}
			}
			//kprintf("Vehicle # %ld is here2.1\n", vehiclenumber);
			if(turndirection == 0){ //Car coming from Direction A wants to make a left, will use locks AB-BC
				kprintf("\nTruck # %ld is approaching the Intersection from Route A to make a left turn (Needs Locks AB-BC).\n",vehiclenumber);
				while(1){
					lock_acquire(mutexCheckVehiclesMakingLeftTurn);
					if(vehiclesMakingLeftTurn == 2){
						lock_release(mutexCheckVehiclesMakingLeftTurn);
						int spl = splhigh();
						thread_sleep(sleepingVehiclesMakingLeftTurn);
						splx(spl);
					}
					else{
						vehiclesMakingLeftTurn++;
						lock_release(mutexCheckVehiclesMakingLeftTurn);
						//kprintf("vehiclesMakingLeftTurn: %d\n",vehiclesMakingLeftTurn);
						break;
					}
				}
				//kprintf("Vehicle # %ld is here2.2\n", vehiclenumber);
				turnleft(vehicledirection, vehiclenumber, vehicletype);
			}
			else{ //Car coming from Direction A wants to make a right, will use locks AB
				kprintf("\nTruck # %ld is approaching the Intersection from Route A to make a right turn (Needs Locks AB).\n",vehiclenumber);
				turnright(vehicledirection,vehiclenumber,vehicletype);
			}
		}
		else if (vehicledirection == 1){ //Car coming from Direction B
			while(1){ //Checks if any vehicle is in Lane A priot to trucks entrance
				lock_acquire(mutexVehiclesInLaneB);
				if(vehiclesInLaneB == 0){
					lock_release(mutexVehiclesInLaneB);
					break;
				}
				else{
					lock_release(mutexVehiclesInLaneB);
					int spl = splhigh();
					thread_sleep(trucksSleepingAtAllLanes);
					splx(spl);
				}
			}
			//kprintf("Vehicle # %ld is here2.3\n", vehiclenumber);
			if(turndirection == 0){ //Car coming from Direction B wants to make a left, will use locks BC-CA
				kprintf("\nTruck # %ld is approaching the Intersection from Route B to make a left turn (Needs Locks BC-CA).\n",vehiclenumber);
					while(1){
					lock_acquire(mutexCheckVehiclesMakingLeftTurn);
					if(vehiclesMakingLeftTurn == 2){
						lock_release(mutexCheckVehiclesMakingLeftTurn);
						int spl = splhigh();
						thread_sleep(sleepingVehiclesMakingLeftTurn);
						splx(spl);
					}
					else{
						vehiclesMakingLeftTurn++;
						lock_release(mutexCheckVehiclesMakingLeftTurn);
						//kprintf("vehiclesMakingLeftTurn: %d\n",vehiclesMakingLeftTurn);
						break;
					}
				}
				//kprintf("Vehicle # %ld is here2.4\n", vehiclenumber);
				turnleft(vehicledirection, vehiclenumber, vehicletype);
			}
			else{ //Car coming from Direction B wants to make a right, will use locks BC
				kprintf("\nTruck # %ld is approaching the Intersection from Route B to make a right turn (Needs Locks BC).\n",vehiclenumber);
				turnright(vehicledirection,vehiclenumber,vehicletype);
			}		
		}
		else{ //Car coming from Direction C
			while(1){ //Checks if any vehicle is in Lane A priot to trucks entrance
				lock_acquire(mutexVehiclesInLaneC);
				if(vehiclesInLaneA == 0){
					lock_release(mutexVehiclesInLaneC);
					break;
				}
				else{
					lock_release(mutexVehiclesInLaneC);
					int spl = splhigh();
					thread_sleep(trucksSleepingAtAllLanes);
					splx(spl);
				}
			}
			//kprintf("Vehicle # %ld is here2.5\n", vehiclenumber);
			if(turndirection == 0){ //Car coming from Direction C wants to make a left, will use locks CA-AB
				kprintf("\nTruck # %ld is approaching the Intersection from Route C to make a left turn (Needs Locks CA-AB).\n",vehiclenumber);	 
				while(1){
					lock_acquire(mutexCheckVehiclesMakingLeftTurn);
					if(vehiclesMakingLeftTurn == 2){
						lock_release(mutexCheckVehiclesMakingLeftTurn);
						int spl = splhigh();
						thread_sleep(sleepingVehiclesMakingLeftTurn);
						splx(spl);
					}
					else{
						vehiclesMakingLeftTurn++;
						lock_release(mutexCheckVehiclesMakingLeftTurn);
						//kprintf("vehiclesMakingLeftTurn: %d\n",vehiclesMakingLeftTurn);
						break;
					}
				}
				//kprintf("Vehicle # %ld is here2.6\n", vehiclenumber);
				turnleft(vehicledirection, vehiclenumber, vehicletype);		
			}
			else{ //Car coming from Direction C wants to make a right, will use locks CA
				kprintf("\nTruck # %ld is approaching the Intersection from Route C to make a right turn (Needs Locks CA).\n",vehiclenumber);	
				turnright(vehicledirection,vehiclenumber,vehicletype);
			}			
		}	
	}
	lock_acquire(mutexMainProcessToExit);
	finishedProcesses++;
	if(finishedProcesses == NVEHICLES){
		lock_release(mutexMainProcessToExit);
		int spl = splhigh();
		thread_wakeup(mainProcessSleepingHere);
		splx(spl);
	}
	else{
		lock_release(mutexMainProcessToExit);
	}
	//kprintf("\nFinished Processes: %d\n",finishedProcesses);
	thread_exit();
}


/*
 * createvehicles()
 *
 * Arguments:
 *      int nargs: unused.
 *      char ** args: unused.
 *
 * Returns:
 *      0 on success.
 *
 * Notes:
 *      Driver code to start up the approachintersection() threads.  You are
 *      free to modiy this code as necessary for your solution.
 */

int
createvehicles(int nargs,
		char ** args)
{
	int index, error;
	AB = lock_create("AB");
	BC = lock_create("BC");
	CA = lock_create("CA");
mutexVehiclesInLaneA = lock_create("mutexVehiclesInLaneA");
mutexVehiclesInLaneB = lock_create("mutexVehiclesInLaneB");
mutexVehiclesInLaneC = lock_create("mutexVehiclesInLaneC");
mutexCheckVehiclesMakingLeftTurn = lock_create("mutexCheckVehiclesMakingLeftTurn");
mutexMainProcessToExit = lock_create("mutexMainProcessToExit");
finishedProcesses = 0;
vehiclesInLaneA = 0;
vehiclesInLaneB = 0;
vehiclesInLaneC = 0;
vehiclesMakingLeftTurn = 0;
	/*
	 * Avoid unused variable warnings.
	 */

	(void) nargs;
	(void) args;

	/*
	 * Start NVEHICLES approachintersection() threads.
	 */

	for (index = 0; index < NVEHICLES; index++) {

		error = thread_fork("approachintersection thread",
				NULL,
				index,
				approachintersection,
				NULL
				);

		/*
		 * panic() on error.
		 */

		if (error) {

			panic("approachintersection: thread_fork failed: %s\n",
					strerror(error)
				 );
		}
	}
	while(1){
		lock_acquire(mutexMainProcessToExit);
		if(finishedProcesses == NVEHICLES){
			lock_release(mutexMainProcessToExit);
			break;
		}
		else{
			lock_release(mutexMainProcessToExit);
			//kprintf("Mains is about to sleep");
			int spl = splhigh();
			thread_sleep(mainProcessSleepingHere);
			splx(spl);
		}
	}
	//kprintf("Main is about to end program");
	return 0;
}
