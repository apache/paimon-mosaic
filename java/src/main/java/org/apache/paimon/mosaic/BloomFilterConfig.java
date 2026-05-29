/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.paimon.mosaic;

import java.util.Objects;

public final class BloomFilterConfig {

    public static final double DEFAULT_FPP = 0.01;

    private final String columnName;
    private final long ndv;
    private final double fpp;

    public BloomFilterConfig(String columnName, long ndv) {
        this(columnName, ndv, DEFAULT_FPP);
    }

    public BloomFilterConfig(String columnName, long ndv, double fpp) {
        this.columnName = Objects.requireNonNull(columnName, "columnName");
        if (ndv < 0) {
            throw new IllegalArgumentException("ndv must be non-negative");
        }
        if (!(fpp > 0.0 && fpp < 1.0)) {
            throw new IllegalArgumentException("fpp must be in (0, 1), got " + fpp);
        }
        this.ndv = ndv;
        this.fpp = fpp;
    }

    public String columnName() { return columnName; }
    public long ndv() { return ndv; }
    public double fpp() { return fpp; }
}
