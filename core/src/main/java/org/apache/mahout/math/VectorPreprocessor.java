package org.apache.mahout.math;

import org.apache.hadoop.conf.Configurable;

/**
 * Hooks into {@link VectorWritable} to provide matrix data preprocessing 
 * capabilities to save on allocating long vectors .
 * 
 * @author Dmitriy
 *
 */
public interface VectorPreprocessor extends Configurable {
	
	/**
	 * called before starting processing a new vector in the writable.
	 * 
	 * @param sequential true if vector is either dense or sequential sparse. 
	 * false if vector formation is non-sequential.
	 * 
	 * @return true if preprocessing of this vector type is supported. 
	 * if preprocessing is not supported, {@link VectorWritable} accumulates 
	 * the vector in the usual way instead of feeding to the preprocessor. 
	 */
	boolean beginVector ( boolean sequential );
	
	/**
	 * called when next vector element becomes available. 
	 * @param index
	 * @param value
	 */
	void onElement ( int index, double value ); 
	
	/**
	 * called for named vectors if vector name is available.
	 * 
	 * @param name
	 */
	void onVectorName ( String name ) ; 
	/**
	 * called after last element is processed.
	 */
	void endVector ();
	

}
