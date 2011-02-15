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

package org.apache.mahout.math.hadoop.stochasticsvd.io;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.Collection;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Misc contributed I/O utils for stochastic svd solvers
 * 
 * @author dmitriy
 * 
 */
public class IOUtil {

  private static final Logger s_log = LoggerFactory.getLogger(IOUtil.class);

  /**
   * make sure to close all sources, log all of the problems occurred, clear
   * <code>closeables</code> (to prevent repeating close attempts), re-throw the
   * last one at the end. Helps resource scope management (e.g. compositions of
   * {@link Closeable}s objects)
   * <P>
   * <p/>
   * Typical pattern:
   * <p/>
   * 
   * <pre>
   *   LinkedList<Closeable> closeables = new LinkedList<Closeable>();
   *   try {
   *      InputStream stream1 = new FileInputStream(...);
   *      closeables.addFirst(stream1);
   *      ...
   *      InputStream streamN = new FileInputStream(...);
   *      closeables.addFirst(streamN);
   *      ...
   *   } finally {
   *      IOUtils.closeAll(closeables);
   *   }
   * </pre>
   * 
   * @param closeables
   *          must be a modifiable collection of {@link Closeable}s
   * @throws IOException
   *           the last exception (if any) of all closed resources
   * 
   * 
   */
  public static void closeAll(Collection<? extends Closeable> closeables)
      throws IOException {
    Throwable lastThr = null;

    for (Closeable closeable : closeables) {
      try {
        closeable.close();
      } catch (Throwable thr) {
        s_log.error(thr.getMessage(), thr);
        lastThr = thr;
      }
    }

    // make sure we don't double-close
    // but that has to be modifiable collection
    closeables.clear();

    if (lastThr != null) {
      if (lastThr instanceof IOException) {
        throw (IOException) lastThr;
      } else if (lastThr instanceof RuntimeException) {
        throw (RuntimeException) lastThr;
      } else if (lastThr instanceof Error) {
        throw (Error) lastThr;
      }
      // should not happen
      else {
        throw (IOException) new IOException("Unexpected exception during close")
            .initCause(lastThr);
      }
    }

  }

  /**
   * a wrapping proxy for interfaces implementing Closeable. it implements
   * two-state fail-fast state pattern which basically has two states: before
   * close and after. any attempt to call resource method after it has been
   * closed would fail.
   * <P>
   * 
   * But it does not to make attempt to wait till the end of current invocations
   * to complete if close() is called, which means it may be possible to
   * actually attempt to invoke close() twice if attempts to close made before
   * any of them had actually completed. Which is why it is fail-fast detection,
   * i.e. no attempt to serialize invocations is made.
   * <P>
   * 
   */

  public static <T extends Closeable> T wrapCloseable(final T delegate,
      Class<T> iface) {
    return iface.cast(Proxy.newProxyInstance(delegate.getClass()
        .getClassLoader(), new Class<?>[] { iface }, new InvocationHandler() {

      private boolean _closedState = false;

      @Override
      public Object invoke(Object proxy, Method method, Object[] args)
          throws Throwable {
        if (_closedState) {
          throw new IOException("attempt to invoke a closed resource.");
        }
        try {
          if (method.equals(s_closeMethod)) {
            _closedState = true;
          }
          return method.invoke(delegate, args);
        } catch (InvocationTargetException exc) {
          throw exc.getTargetException();
        }
      }
    }));
  }

  private static final Method s_closeMethod;

  static {
    try {
      s_closeMethod = Closeable.class.getMethod("close");
    } catch (NoSuchMethodException exc) {
      // should not happen
      throw new RuntimeException(exc);
    }
  }

  /**
   * for temporary files, a file may be considered as a {@link Closeable} too,
   * where file is wiped on close and thus the disk resource is released
   * ('closed').
   * 
   * @author dmitriy
   * 
   */
  public static class DeleteFileOnClose implements Closeable {

    private File m_file;

    public DeleteFileOnClose(File file) {
      m_file = file;
    }

    @Override
    public void close() throws IOException {
      if (m_file.isFile())
        m_file.delete();
    }

  }

  /**
   * convienience wrapper
   * 
   * @param <T>
   *          the cloning type
   * @param src
   *          the object being cloned
   * @return a clone of the object, or <code>src</code> if clone is not
   *         supported.
   */
  public static <T> T tryClone(T src) {
    try {
      Method cloneMethod = src.getClass().getMethod("clone");
      @SuppressWarnings("unchecked")
      T result = (T) cloneMethod.invoke(src);
      return result;
    } catch (NoSuchMethodException exc) {
      return src;
    } catch (InvocationTargetException exc) {
      if (exc.getTargetException() instanceof CloneNotSupportedException) {
        return src;
      } else {
        throw new RuntimeException(exc.getTargetException());
      }
    } catch (IllegalAccessException exc) {
      return src;
    }
  }

}
