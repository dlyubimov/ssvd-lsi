/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math.hadoop.stochasticsvd;

import java.util.Random;

import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

/**
 * simplistic implementation for Omega matrix 
 * 
 * @author dmitriy
 *
 */
public class Omega {

	private long 		m_seed;
	private Random 		m_rnd=new Random();
	private int 		m_columns; 
	
	public Omega(long seed, int k, int p) {
		super();
		m_seed=seed;
		m_columns=k+p;
		
		
	}

	
	/**
	 * compute YRow=ARow*Omega.
	 * 
	 * @param ARow row of matrix A (size n)
	 * @param YRow row of matrix Y (result) must be pre-allocated to size of (k+p)
	 */
	public void computeYRow(
			Vector ARow,
			Vector YRow
			) { 
		if ( YRow.size()< m_columns )
			throw new RuntimeException ( "YRow must have size at least k+p");
		
		
		
		for ( int i = 0; i < m_columns; i++ ) { 
			// set rnd to column seed
			// TODO: perhaps there is a much better way to do this.
			m_rnd.setSeed(m_seed+i);
			int n = ARow.size();
			double y_i=0;
			if ( ARow instanceof SequentialAccessSparseVector ) 
			    for (Element el:ARow ) 
			        y_i+=el.get()*(m_rnd.nextGaussian());
			       
			else for ( int j = 0; j < n; j++ ) 
				y_i+=ARow.getQuick(j)*(m_rnd.nextGaussian());
			YRow.set(i,y_i);
		}
	}
	
	
    public static long murmur64 ( byte[] val, int offset, int len, long seed ) {
        
        long m= 0xc6a4a7935bd1e995L;
        int r = 47; 
        long h = seed ^ (len * m );
        
        int lt = len >>>3;
        for ( int i =0; i < lt; i++, offset += 8 ) {
            long k =0;
            for ( int j= 0; j < 8; j++ ) {k <<=8; k |= val[offset+j]&0xff; }

            k*=m;
            k^= k>>>r;
            k*=m;
            
            h^=k;
            h*=m;
        }
        long k = 0;
        
        if ( offset < len ) { 
            for ( ; offset < len; offset ++ ) { k<<=8; k|= val[offset]&0xff; }
            h^=k; 
            h*=m;
        }
        
        h^=h>>>r;
        h*=m;
        h^= h>>>r;
        return h;
        
    }

}
